#!/usr/bin/env bash
# Impair only the brain-to-brain team traffic of the sim with tc/netem. Needs root (uses `sudo -n`).
#   scripts/netem.sh up                       create bridge br-sim and namespaces ss-r1..ss-rN (N=$SS_ROBOTS, default 10)
#   scripts/netem.sh apply <profile> [robots] profile: good|rcap_bad|terrible|blackout[:SECONDS]; robots: all (default) or "1,3,4"
#   scripts/netem.sh status                   print the active qdiscs/filters
#   scripts/netem.sh down                     remove everything this script created
# Run a brain in robot N's namespace with:  sudo -n ip netns exec ss-rN sudo -n -u $USER env ... <cmd>   (see docs/netem.md)
# SS_NETEM_INGRESS=1: impair on each receiver instead (ifb ssi<N> in the namespace, mirred redirect of ssv<N> ingress) so a
#   broadcast is lost independently per receiver; the profile of robot N then describes N's *receiving* link (see docs/netem.md).
# Only objects named ss-* / br-sim / ssv* / ssh* / ssi* are ever touched.
set -euo pipefail
N="${SS_ROBOTS:-10}"
BR=br-sim
SUBNET=10.77.0            # bridge 10.77.0.1/24, robot i = 10.77.0.(10+i)
TEAM_PORTS="${SS_TEAM_PORTS:-10001 10131}"   # UDP dst ports of team comm = 10000 + team number (away=1, home=131)
STATE=/tmp/ss-netem-state
INGRESS="${SS_NETEM_INGRESS:-0}"
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -n"
inns() { $SUDO ip netns exec "ss-r$1" "${@:2}"; }

profile_args() {  # -> netem arguments
  case "$1" in
    good)     echo "delay 5ms" ;;
    rcap_bad) echo "delay 150ms 120ms distribution normal loss gemodel 5% 20% 75% 0.1%" ;;
    terrible) echo "delay 600ms 200ms loss 40%" ;;
    blackout) echo "loss 100%" ;;
    *) echo "unknown profile '$1' (good|rcap_bad|terrible|blackout[:SECONDS])" >&2; return 1 ;;
  esac
}
# gemodel p r 1-h 1-k: p=5% good->bad, r=20% bad->good (mean bad sojourn 5 pkts), 75% loss in bad, 0.1% in good
# => stationary bad fraction p/(p+r)=20%, mean loss ~ 0.2*0.75+0.8*0.001 = 15.1%  (see docs/netem.md)

robot_list() { if [ "${1:-all}" = all ]; then seq 1 "$N"; else echo "$1" | tr ',' ' '; fi; }

# prio qdisc whose bands all default to band 1 (untouched traffic); netem sits on band 3, u32 filters send team UDP there.
setup_dev() {  # <runner...> dev netem-args
  local dev="$2" args="$3"; local -a run=($1)
  "${run[@]}" tc qdisc replace dev "$dev" root handle 1: prio bands 3 priomap 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  "${run[@]}" tc qdisc replace dev "$dev" parent 1:3 handle 30: netem $args
  "${run[@]}" tc filter del dev "$dev" parent 1: 2>/dev/null || true
  local p
  for p in $TEAM_PORTS; do
    "${run[@]}" tc filter add dev "$dev" parent 1: protocol ip prio 1 u32 \
      match ip protocol 17 0xff match ip dport "$p" 0xffff flowid 1:3
  done
}

cur_profile() { cat "$STATE/r$1" 2>/dev/null || echo none; }

# Receiver-side impairment: ssv<N> ingress -> (team UDP only) mirred redirect -> ifb ssi<N>, whose egress carries the profile.
setup_ingress() {  # <robot> netem-args
  local i="$1" args="$2" p
  inns "$i" ip link show "ssi$i" >/dev/null 2>&1 || inns "$i" ip link add "ssi$i" type ifb
  inns "$i" ip link set "ssi$i" up
  inns "$i" tc qdisc replace dev "ssv$i" handle ffff: ingress
  inns "$i" tc filter del dev "ssv$i" parent ffff: 2>/dev/null || true
  for p in $TEAM_PORTS; do
    inns "$i" tc filter add dev "ssv$i" parent ffff: protocol ip prio 1 u32 \
      match ip protocol 17 0xff match ip dport "$p" 0xffff \
      action mirred egress redirect dev "ssi$i"
  done
  setup_dev "$SUDO ip netns exec ss-r$i" "ssi$i" "$args"
}

apply_one() {  # <robot> <profile>
  local i="$1" prof="$2" base="${2%%:*}" args
  args="$(profile_args "$base")"
  if [ "$INGRESS" = 1 ]; then
    setup_ingress "$i" "$args"
    $SUDO tc qdisc del dev "ssh$i" root 2>/dev/null || true
    if [ "$base" = blackout ]; then                              # a blackout also cuts the uplink
      setup_dev "$SUDO ip netns exec ss-r$i" "ssv$i" "$args"
    else
      inns "$i" tc qdisc del dev "ssv$i" root 2>/dev/null || true
    fi
    return
  fi
  setup_dev "$SUDO ip netns exec ss-r$i" "ssv$i" "$args"         # robot -> router (uplink), inside the namespace
  if [ "$base" = blackout ]; then                                # a blackout cuts the downlink too
    setup_dev "$SUDO" "ssh$i" "$args"
  else
    $SUDO tc qdisc del dev "ssh$i" root 2>/dev/null || true
  fi
}

cmd_up() {
  mkdir -p "$STATE"
  if ! ip link show "$BR" >/dev/null 2>&1; then
    $SUDO ip link add "$BR" type bridge
  fi
  $SUDO ip addr replace "$SUBNET.1/24" dev "$BR"; $SUDO ip link set "$BR" up
  local i
  for i in $(seq 1 "$N"); do
    ip netns list | grep -qx "ss-r$i\( .*\)\?" || $SUDO ip netns add "ss-r$i"
    if ! ip link show "ssh$i" >/dev/null 2>&1; then
      $SUDO ip link add "ssh$i" type veth peer name "ssv$i" netns "ss-r$i"
      $SUDO ip link set "ssh$i" master "$BR"
    fi
    $SUDO ip link set "ssh$i" up
    inns "$i" ip link set lo up
    inns "$i" ip addr replace "$SUBNET.$((10 + i))/24" dev "ssv$i"
    inns "$i" ip link set "ssv$i" up
    inns "$i" ip link set "ssv$i" multicast on
    inns "$i" ip route replace default via "$SUBNET.1" dev "ssv$i"
    inns "$i" ip route replace 224.0.0.0/4 dev "ssv$i"
    apply_one "$i" good; echo good > "$STATE/r$i"
  done
  echo "up: $BR $SUBNET.1/24, namespaces ss-r1..ss-r$N (profile good)"
}

cmd_apply() {
  local prof="${1:?profile}" r base dur=""
  base="${prof%%:*}"; [ "$base" != "$prof" ] && dur="${prof#*:}"
  profile_args "$base" >/dev/null
  [ -z "$dur" ] || [ "$base" = blackout ] || { echo "duration only applies to blackout" >&2; exit 2; }
  mkdir -p "$STATE"
  for r in $(robot_list "${2:-all}"); do
    ip netns list | grep -q "^ss-r$r\b" || { echo "ss-r$r missing: run '$0 up' first" >&2; exit 1; }
    local prev; prev="$(cur_profile "$r")"
    apply_one "$r" "$base"
    if [ "$base" = blackout ]; then
      [ "${prev%%:*}" = blackout ] && prev="$(cat "$STATE/r$r.prev" 2>/dev/null || echo good)" || echo "$prev" > "$STATE/r$r.prev"
    fi
    echo "$base" > "$STATE/r$r"
    echo "ss-r$r: $base"
  done
  if [ -n "$dur" ]; then   # revert each robot to the profile it had before the blackout
    local list; list="$(robot_list "${2:-all}" | tr '\n' ',' | sed 's/,$//')"
    ( sleep "$dur"; for r in $(robot_list "$list"); do
        [ "$(cur_profile "$r")" = blackout ] && "$0" apply "$(cat "$STATE/r$r.prev" 2>/dev/null || echo good)" "$r" >/dev/null
      done ) >/dev/null 2>&1 &
    echo "blackout reverts in ${dur}s"
  fi
}

cmd_status() {
  local i
  echo "== $BR"; ip -br addr show "$BR" 2>/dev/null || echo "(absent)"
  for i in $(seq 1 "$N"); do
    ip netns list | grep -q "^ss-r$i\b" || continue
    echo "== ss-r$i (profile $(cur_profile "$i")) uplink ssv$i"
    inns "$i" tc qdisc show dev "ssv$i"; [ "$INGRESS" = 1 ] && inns "$i" tc qdisc show dev "ssi$i"; inns "$i" tc filter show dev "ssv$i" | grep -c "flowid 1:3" | sed 's/^/  team filters: /'
    echo "   downlink ssh$i:"; $SUDO tc qdisc show dev "ssh$i" | sed 's/^/   /'
  done
}

cmd_down() {
  local i
  for i in $(seq 1 "$N"); do
    ip link show "ssh$i" >/dev/null 2>&1 && $SUDO ip link del "ssh$i"   # removes the peer too
    ip netns list | grep -q "^ss-r$i\b" && $SUDO ip netns del "ss-r$i"
  done
  ip link show "$BR" >/dev/null 2>&1 && $SUDO ip link del "$BR"
  rm -rf "$STATE"
  echo "down"
  return 0
}

case "${1:-}" in
  up) cmd_up ;; down) cmd_down ;; apply) shift; cmd_apply "$@" ;; status) cmd_status ;;
  *) sed -n 2,7p "$0"; exit 2 ;;
esac
