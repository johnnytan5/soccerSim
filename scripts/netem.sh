#!/usr/bin/env bash
# Impair only the brain-to-brain team traffic of the sim with tc/netem. Needs root (uses `sudo -n`).
#   scripts/netem.sh up                       create bridge br-sim and namespaces ss-r1..ss-rN (N=$SS_ROBOTS, default 10; slot: see SS_SLOT)
#   scripts/netem.sh apply <profile> [robots] profile: good|rcap_bad|terrible|blackout[:SECONDS]; robots: all (default) or "1,3,4"
#   scripts/netem.sh status                   print the active qdiscs/filters
#   scripts/netem.sh down                     remove everything this script created
# Run a brain in robot N's namespace with:  sudo -n ip netns exec ss-rN sudo -n -u $USER env ... <cmd>   (see docs/netem.md)
# SS_NETEM_INGRESS=1: impair on each receiver instead (ifb ssi<N> in the namespace, mirred redirect of ssv<N> ingress) so a
#   broadcast is lost independently per receiver; the profile of robot N then describes N's *receiving* link (see docs/netem.md).
# SS_SLOT=<n> (0..9; default empty = the names above): a parallel, independent copy of everything: namespaces ss<n>-r1..,
#   bridge br-sim<n> = 10.77.<n+1>.1/24, veths ssh<n>_<i> / ssv<n>_<i>, ifb ssi<n>_<i>, state /tmp/ss-netem-state<n>.
#   `netem.sh names` prints the resulting names (no root needed). See docs/netem.md.
# Profiles: good | rcap_bad | terrible | blackout[:SECONDS] | lat<ms>_loss<pct> (delay ms +-0.4*ms normal, bursty loss with mean pct%).
# Only objects named ss* / br-sim* are ever touched.
set -euo pipefail
N="${SS_ROBOTS:-10}"
SLOT="${SS_SLOT:-}"
case "$SLOT" in
  "") BR=br-sim; SUBNET=10.77.0; NSP=ss-r; HP=ssh; VP=ssv; IP=ssi; STATE=/tmp/ss-netem-state ;;   # bridge 10.77.0.1/24, robot i = .(10+i)
  [0-9]) BR="br-sim$SLOT"; SUBNET="10.77.$((SLOT + 1))"; NSP="ss$SLOT-r"; HP="ssh${SLOT}_"; VP="ssv${SLOT}_"; IP="ssi${SLOT}_"
         STATE="/tmp/ss-netem-state$SLOT" ;;
  *) echo "SS_SLOT must be empty or a single digit 0-9, not '$SLOT'" >&2; exit 2 ;;
esac
TEAM_PORTS="${SS_TEAM_PORTS:-10001 10131}"   # UDP dst ports of team comm = 10000 + team number (away=1, home=131)
INGRESS="${SS_NETEM_INGRESS:-0}"
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -n"
inns() { $SUDO ip netns exec "$NSP$1" "${@:2}"; }

profile_args() {  # -> netem arguments
  case "$1" in
    good)     echo "delay 5ms" ;;
    rcap_bad) echo "delay 150ms 120ms distribution normal loss gemodel 5% 20% 75% 0.1%" ;;
    terrible) echo "delay 600ms 200ms loss 40%" ;;
    blackout) echo "loss 100%" ;;
    lat*_loss*) custom_args "$1" ;;
    *) echo "unknown profile '$1' (good|rcap_bad|terrible|blackout[:SECONDS]|lat<ms>_loss<pct>)" >&2; return 1 ;;
  esac
}
# lat<ms>_loss<pct>: delay = ms with jitter 0.4*ms (normal); loss = Gilbert-Elliott with mean pct%, mean bad burst 1/r = 3.3 pkts
# (r = 0.3), loss in the good state 0 and lb in the bad state (75%, or 100% above 60% mean loss). The stationary bad fraction is
# pi = L/lb, so p = r*pi/(1-pi). lat0_loss0 = no impairment (netem pass-through). See docs/netem.md.
custom_args() {
  local ms pct
  [[ "$1" =~ ^lat([0-9]+)_loss([0-9]+(\.[0-9]+)?)$ ]] || { echo "bad profile '$1': lat<ms>_loss<pct>, e.g. lat100_loss5" >&2; return 1; }
  ms="${BASH_REMATCH[1]}"; pct="${BASH_REMATCH[2]}"
  awk -v ms="$ms" -v pct="$pct" 'BEGIN {
    if (pct > 100) { print "loss must be <= 100" > "/dev/stderr"; exit 1 }
    out = ""
    if (ms > 0) out = sprintf("delay %dms %gms distribution normal", ms, 0.4 * ms)
    if (pct >= 100) out = out " loss 100%"
    else if (pct > 0) {
      L = pct / 100; lb = (pct <= 60) ? 0.75 : 1.0; r = 0.3; pi = L / lb
      pgb = r * pi / (1 - pi)
      if (pgb > 1) { pgb = 1; r = (1 - pi) / pi }      # very high loss: p capped at 1, the bad burst shortens instead
      out = out sprintf(" loss gemodel %.4f%% %.4f%% %.4f%% 0%%", 100 * pgb, 100 * r, 100 * lb)
    }
    if (out == "") out = "limit 1000"
    sub(/^ /, "", out); print out
  }'
}
# gemodel p r 1-h 1-k: p=5% good->bad, r=20% bad->good (mean bad sojourn 5 pkts), 75% loss in bad, 0.1% in good
# => stationary bad fraction p/(p+r)=20%, mean loss ~ 0.2*0.75+0.8*0.001 = 15.1%  (see docs/netem.md)

robot_list() { if [ "${1:-all}" = all ]; then seq 1 "$N"; else echo "$1" | tr ',' ' '; fi; }

# prio qdisc whose bands all default to band 1 (untouched traffic); netem sits on band 3, u32 filters send team UDP there.
# Fast path: when the structure from an earlier apply is in place only the netem leaf is changed (`tc qdisc change`), the
# u32 filters are never touched, so team packets cannot bypass netem during a blackout / revert. Otherwise the whole
# structure is built with one `tc -batch` per device.
setup_dev() {  # <runner...> dev netem-args
  local dev="$2" args="$3"; local -a run=($1)
  if "${run[@]}" tc qdisc show dev "$dev" 2>/dev/null | grep -q "qdisc netem 30: parent 1:3" \
     && "${run[@]}" tc filter show dev "$dev" parent 1: 2>/dev/null | grep -q "flowid 1:3" \
     && "${run[@]}" tc qdisc change dev "$dev" parent 1:3 handle 30: netem $args 2>/dev/null; then
    return 0
  fi
  local p
  {
    echo "qdisc replace dev $dev root handle 1: prio bands 3 priomap 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    echo "qdisc replace dev $dev parent 1:3 handle 30: netem $args"
    echo "filter del dev $dev parent 1:"
    for p in $TEAM_PORTS; do
      echo "filter add dev $dev parent 1: protocol ip prio 1 u32 match ip protocol 17 0xff match ip dport $p 0xffff flowid 1:3"
    done
  } | "${run[@]}" tc -force -batch -
}

cur_profile() { cat "$STATE/r$1" 2>/dev/null || echo none; }

# Receiver-side impairment: ssv<N> ingress -> (team UDP only) mirred redirect -> ifb ssi<N>, whose egress carries the profile.
setup_ingress() {  # <robot> netem-args
  local i="$1" args="$2" p
  inns "$i" ip link show "$IP$i" >/dev/null 2>&1 || inns "$i" ip link add "$IP$i" type ifb
  inns "$i" ip link set "$IP$i" up
  if ! { inns "$i" tc qdisc show dev "$VP$i" | grep -q "ingress ffff:" \
         && inns "$i" tc filter show dev "$VP$i" parent ffff: | grep -q mirred; }; then   # only (re)build when missing
    inns "$i" tc qdisc replace dev "$VP$i" handle ffff: ingress
    inns "$i" tc filter del dev "$VP$i" parent ffff: 2>/dev/null || true
    for p in $TEAM_PORTS; do
      inns "$i" tc filter add dev "$VP$i" parent ffff: protocol ip prio 1 u32 \
        match ip protocol 17 0xff match ip dport "$p" 0xffff \
        action mirred egress redirect dev "$IP$i"
    done
  fi
  setup_dev "$SUDO ip netns exec $NSP$i" "$IP$i" "$args"
}

apply_one() {  # <robot> <profile>
  local i="$1" prof="$2" base="${2%%:*}" args
  args="$(profile_args "$base")"
  if [ "$INGRESS" = 1 ]; then
    setup_ingress "$i" "$args"
    $SUDO tc qdisc del dev "$HP$i" root 2>/dev/null || true
    if [ "$base" = blackout ]; then                              # a blackout also cuts the uplink
      setup_dev "$SUDO ip netns exec $NSP$i" "$VP$i" "$args"
    else
      inns "$i" tc qdisc del dev "$VP$i" root 2>/dev/null || true
    fi
    return
  fi
  setup_dev "$SUDO ip netns exec $NSP$i" "$VP$i" "$args"         # robot -> router (uplink), inside the namespace
  if [ "$base" = blackout ]; then                                # a blackout cuts the downlink too
    setup_dev "$SUDO" "$HP$i" "$args"
  else
    $SUDO tc qdisc del dev "$HP$i" root 2>/dev/null || true
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
    ip netns list | grep -qx "$NSP$i\( .*\)\?" || $SUDO ip netns add "$NSP$i"
    if ! ip link show "$HP$i" >/dev/null 2>&1; then
      $SUDO ip link add "$HP$i" type veth peer name "$VP$i" netns "$NSP$i"
      $SUDO ip link set "$HP$i" master "$BR"
    fi
    $SUDO ip link set "$HP$i" up
    inns "$i" ip link set lo up
    inns "$i" ip addr replace "$SUBNET.$((10 + i))/24" dev "$VP$i"
    inns "$i" ip link set "$VP$i" up
    inns "$i" ip link set "$VP$i" multicast on
    inns "$i" ip route replace default via "$SUBNET.1" dev "$VP$i"
    inns "$i" ip route replace 224.0.0.0/4 dev "$VP$i"
    apply_one "$i" good; echo good > "$STATE/r$i"
  done
  echo "up: $BR $SUBNET.1/24, namespaces ${NSP}1..$NSP$N (profile good)"
}

cmd_apply() {
  local prof="${1:?profile}" r base dur=""
  base="${prof%%:*}"; [ "$base" != "$prof" ] && dur="${prof#*:}"
  profile_args "$base" >/dev/null
  [ -z "$dur" ] || [ "$base" = blackout ] || { echo "duration only applies to blackout" >&2; exit 2; }
  mkdir -p "$STATE"
  local -a pids=(); local rc=0 p
  # every robot in parallel (each does ~10 tc calls): a transition then takes about one robot's time, not N times it
  for r in $(robot_list "${2:-all}"); do
    ip netns list | grep -q "^$NSP$r\b" || { echo "$NSP$r missing: run '$0 up' first" >&2; exit 1; }
    (
      prev="$(cur_profile "$r")"
      apply_one "$r" "$base"
      if [ "$base" = blackout ]; then
        [ "${prev%%:*}" = blackout ] && prev="$(cat "$STATE/r$r.prev" 2>/dev/null || echo good)" || echo "$prev" > "$STATE/r$r.prev"
      fi
      echo "$base" > "$STATE/r$r"
      echo "$NSP$r: $base"
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" = 0 ] || { echo "apply failed on at least one robot" >&2; exit 1; }
  if [ -n "$dur" ]; then   # revert each robot to the profile it had before the blackout
    local list; list="$(robot_list "${2:-all}" | tr '\n' ',' | sed 's/,$//')"
    ( sleep "$dur"; for r in $(robot_list "$list"); do
        [ "$(cur_profile "$r")" = blackout ] && "$0" apply "$(cat "$STATE/r$r.prev" 2>/dev/null || echo good)" "$r" >/dev/null &
      done; wait ) >/dev/null 2>&1 &
    echo "blackout reverts in ${dur}s"
  fi
}

cmd_status() {
  local i
  echo "== $BR"; ip -br addr show "$BR" 2>/dev/null || echo "(absent)"
  for i in $(seq 1 "$N"); do
    ip netns list | grep -q "^$NSP$i\b" || continue
    echo "== $NSP$i (profile $(cur_profile "$i")) uplink $VP$i"
    inns "$i" tc qdisc show dev "$VP$i"; [ "$INGRESS" = 1 ] && inns "$i" tc qdisc show dev "$IP$i"; inns "$i" tc filter show dev "$VP$i" | grep -c "flowid 1:3" | sed 's/^/  team filters: /'
    echo "   downlink $HP$i:"; $SUDO tc qdisc show dev "$HP$i" | sed 's/^/   /'
  done
}

cmd_down() {
  local i
  for i in $(seq 1 "$N"); do
    ip link show "$HP$i" >/dev/null 2>&1 && $SUDO ip link del "$HP$i"   # removes the peer too
    ip netns list | grep -q "^$NSP$i\b" && $SUDO ip netns del "$NSP$i"
  done
  ip link show "$BR" >/dev/null 2>&1 && $SUDO ip link del "$BR"
  rm -rf "$STATE"
  echo "down"
  return 0
}

cmd_names() {  # robot 1's names, for tests and scripts (no root needed)
  echo "bridge=$BR subnet=$SUBNET.0/24 gateway=$SUBNET.1 broadcast=$SUBNET.255 ns1=${NSP}1 host1=${HP}1 peer1=${VP}1 ifb1=${IP}1 state=$STATE"
}

case "${1:-}" in
  names) cmd_names ;;
  up) cmd_up ;; down) cmd_down ;; apply) shift; cmd_apply "$@" ;; status) cmd_status ;;
  *) sed -n 2,7p "$0"; exit 2 ;;
esac
