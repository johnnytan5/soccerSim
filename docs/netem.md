# Team-network impairment (netem)

`scripts/netem.sh` reproduces a bad competition router for the brain-to-brain team channel only. Sensors,
commands and DDS between brains and the sim's `match_node` are not touched.

```
scripts/netem.sh up                         # bridge br-sim + namespaces ss-r1..ss-r10 (SS_ROBOTS=N), profile "good"
scripts/netem.sh apply rcap_bad             # every robot
scripts/netem.sh apply terrible 1,3         # robots 1 and 3 only
scripts/netem.sh apply blackout:10 2        # robot 2 cut off for 10 s, then back to its previous profile
scripts/netem.sh status
scripts/netem.sh down
```

Idempotent: `up` and `apply` can be repeated (`tc ... replace`, existing objects reused). Needs root for `ip netns`, `ip link`
and `tc`; the script calls `sudo -n`, so passwordless sudo is required (or run it as root). It only creates or deletes
`br-sim`, `ssh<N>` / `ssv<N>` veths and `ss-r<N>` namespaces.

## Approach and why

Team traffic: every brain sends its state and discovery packets as UDP to the directed subnet broadcast on
destination port `10000 + team` (home 131 -> 10131, away 1 -> 10001), from source port `30000 + team*MAX_NUM_PLAYERS + player`
(`brain_communication.cpp`); it listens on the same destination port with `SO_REUSEADDR`.

Rejected: one namespace with `tc` on `lo`. Measured in a throwaway namespace (netem 100 ms on both `lo` and the NIC): a
broadcast to the subnet broadcast address arrived in 0.1 ms, i.e. the local copy of a broadcast is looped back inside the IP
layer and never goes through any qdisc. Only unicast to 127.0.0.1 went through `lo` (+100 ms). On the real machine the
broadcast leaves via the interface routing to `game_control_ip` (eth0 here), which must not be touched anyway.

Chosen: one network namespace per robot, joined by veth pairs to a bridge `br-sim` (10.77.0.1/24, robot i = 10.77.0.(10+i)),
so the broadcast really crosses a link. Impairment is `tc` egress on the robot's veth inside its namespace: a `prio` qdisc
whose bands all default to band 1 (untouched traffic), `netem` on band 3, and `u32` filters that send only UDP with
destination port in `SS_TEAM_PORTS` (default `10001 10131`) to band 3. Everything else (DDS, other UDP) takes band 1 and
is unimpaired. Per-robot impairment comes free (one qdisc per namespace). Whole team = apply to that team's robots.

Brains and anything else that must be impaired have to run inside their namespace. The launcher does this (see "Launcher" below) with
`scripts/ns_exec.sh ss-rN <cmd>`; set `game_control_ip` (or
`communication.discovery_address`) to `10.77.0.255` so the broadcast goes out the veth.

DDS across namespaces: verified under `pixi run`: `ros2 topic pub` in `ss-r3` (domain 71) was received by `ros2 topic echo`
in the root namespace (default RMW, discovery works over br-sim with the multicast route that `up` installs).

## Profiles

| Profile | netem | Nominal |
|---|---|---|
| good | `delay 5ms` | 5 ms, no loss |
| rcap_bad | `delay 150ms 120ms distribution normal loss gemodel 5% 20% 75% 0.1%` | ~150 ms +-120 ms, bursty loss ~15% |
| terrible | `delay 600ms 200ms loss 40%` | 600 +-200 ms, 40% |
| blackout[:T] | `loss 100%` (both directions) | total loss; after T seconds returns to the profile it replaced |
| `lat<ms>_loss<pct>` | custom, see below | `ms` delay, jitter 0.4 ms (normal), bursty loss with mean `pct` % |

Gilbert-Elliott: P(good->bad)=5%, P(bad->good)=20% (mean burst ~5 packets), 75% loss in the bad state, 0.1% in good;
stationary loss 0.2*0.75+0.8*0.001 = 15.1%. Optionally add `rate 2mbit` to the rcap_bad line in `profile_args`.

### Custom profiles `lat<ms>_loss<pct>`

`scripts/netem.sh apply lat100_loss5` (also `lat150_loss2.5`; usable in the launcher's network profile and in
`batch_matches --profiles`). Delay = `ms` with jitter `0.4*ms` and `distribution normal`. Loss is Gilbert-Elliott with the
parameters derived from the target mean loss L (`custom_args` in netem.sh):

- loss in the good state 0, in the bad state `lb` = 75% (100% when pct > 60);
- mean bad burst 1/r with r = 30% (3.3 packets);
- stationary bad fraction `pi = L / lb`, so `p = r * pi / (1 - pi)`; mean loss = `pi * lb` = L exactly. (With lb = 100% this is the
  formula `p = L r / (1 - L)`; with 75% in the bad state `pi` must be L/0.75, so that formula would undershoot.)
- if p would exceed 1 (L above ~ 70%), p = 1 and r shrinks instead.

Examples: `lat100_loss5` = `delay 100ms 40ms distribution normal loss gemodel 2.1429% 30% 75% 0%`; `lat150_loss15` =
`... loss gemodel 7.5% 30% 75% 0%`. `lat0_loss0` = no impairment (`netem limit 1000`, a pure pass-through); `lat<ms>_loss0` only
delays, `lat0_loss<pct>` only loses. Like the others, only the team ports are impaired.

Measured with `scripts/netem_probe.py` (the method below: 64 B probes at 200 pkt/s, ss-r1 broadcasts to ss-r2, 6000 probes per
row, applied on robot 1's uplink; run in slot 7). Delay uses CLOCK_MONOTONIC: this WSL's wall clock steps by seconds during a
run, which made an early attempt with `time.time()` show 23 s outliers.

| Profile | Loss nominal / measured | Mean loss burst | Delay nominal (mean, sd) | Measured mean / sd / p99 / max |
|---|---|---|---|---|
| lat0_loss0 (1000 pkts) | 0 / 0.0% | - | none | 0.1 / 0.0 / - / 0.2 ms |
| lat50_loss5 | 5 / 5.07% | 1.96 pkts | 50, 20 ms | 50.3 / 19.7 / 97.6 / 120.5 ms |
| lat100_loss15 | 15 / 14.48% | 2.09 pkts | 100, 40 ms | 100.5 / 39.7 / 192.4 / 241.0 ms |
| lat200_loss30 | 30 / 28.2% | 2.05 pkts | 200, 80 ms | 201.7 / 80.3 / 387.1 / 481.9 ms |

Loss is within ~2 points of nominal (the Markov chain's run-to-run spread over 6000 probes is about that large). Delay mean
and sd match; the clamping of negative normal draws is negligible at jitter = 0.4 ms (2.5 sigma).

## Parallel slots (`SS_SLOT`)

`SS_SLOT=<n>` (one digit 0-9; default empty = everything above, unchanged) gives an independent copy of the whole setup, so
several matches can run at once (`batch_matches --parallel K`, slots 0..K-1):

| | slot empty | slot n |
|---|---|---|
| namespaces | `ss-r1..` | `ss<n>-r1..` |
| bridge | `br-sim` 10.77.0.1/24 | `br-sim<n>` 10.77.<n+1>.1/24 |
| robot i | 10.77.0.(10+i) | 10.77.<n+1>.(10+i) |
| veth host / peer / ifb | `ssh<i>` `ssv<i>` `ssi<i>` | `ssh<n>_<i>` `ssv<n>_<i>` `ssi<n>_<i>` |
| state dir | /tmp/ss-netem-state | /tmp/ss-netem-state<n> |

All names are <= 15 characters and start with `ss` or `br-sim`, so the "only ss* / br-sim* is touched" rule holds.
`scripts/netem.sh names` prints them without root; `soccer_launcher.netns.slot_names` computes the same (a test compares the two).
The launcher side: `LaunchConfig.netns_slot` or `$SS_SLOT`; the brains get `ns_prefix:=ss<n>-r`, `game_control_ip` = the slot's
gateway and the slot's broadcast as `communication.discovery_address`. The bridge's team-packet counter binds to the slot's
broadcast address (`match_node --packet-bind`) so two bridges do not count each other's packets. Slots are independent of
`SS_ROBOTS` (per-slot namespaces 1..N).

## Measured (UDP probes 64 B, 200 pkt/s, sender port 30001 in ss-r1 to broadcast 10.77.0.255:10131, receiver in ss-r2)

| Profile | Loss (nominal) | Delay mean / sd / max |
|---|---|---|
| good | 0.0% (0) | 5.1 / 0.1 ms |
| rcap_bad (4000 pkts) | 14.5% (15.1%), mean burst 2.6 pkts | 159 / 111 / 573 ms |
| terrible (2000 pkts) | 38.5% (40%) | 608 / 199 ms |

A non-team port (20000) on the same path: 0 loss, 0.1 ms in every profile. A third robot (ss-r3) sending while ss-r1 is
rcap_bad: 0 loss, 5 ms. Unicast to a peer IP on a team port is impaired too. Blackout:4 on ss-r1 cut its packets for
~4 s, then it returned to rcap_bad.

Note on rcap_bad delay: the normal distribution gives negative delays about 10% of the time; netem clamps them to 0, so the
mean is a little above 150 ms and ~10% of packets are near-instant. Jittered delay also reorders packets.

## Limitations

- netem is a queue, not WiFi: no airtime contention between robots, no rate collapse or retries, no hidden-node effects, and
  loss/delay are independent of packet count or channel load. Add `rate` only as a crude stand-in.
- Impairment is sender-side (robot to router) for all profiles except blackout, which also cuts the downlink; so a
  robot-to-robot packet is impaired once, and a robot's reception is not degraded unless the sender is. Receive-side
  degradation of a single robot would need an ifb or the `ssh<N>` egress.
- Broadcast fan-out in the bridge copies the packet to each receiver after the sender's netem, so losses are common to all
  receivers (a real router also loses per receiver).
- Team ports are matched by destination port only. DDS domains 9 to 11 use UDP ports around 10000 and could collide; the
  default domain base 70 is fine. Change the ports with `SS_TEAM_PORTS`.
- `apply` runs the robots in parallel and, once a device has the prio + u32 structure, only does `tc qdisc change` on the
  netem leaf, so filters are never removed during a profile change (no window in which team packets bypass netem). The batch
  runner records a blackout as start (before the apply) / apply_s / revert_begin (before the revert) / end (after it) and the
  metrics treat the two ramps as neither blackout nor outside (see docs/batch_matches.md).
- The blackout revert timer is a background shell job: it does not survive `down` or a reboot.

## Launcher (network profile)

Config fields (Control Centre, Match group): `network_profile` (`none` default = no namespaces, `good`, `rcap_bad`, `terrible`,
`blackout[:T]`), `network_overrides` (`home3=terrible,away2=blackout:10`; team + player id) and `network_ingress`. For a
profile other than none the launcher: checks `sudo -n tc`, `sudo -n ip netns`, `sch_netem` (and `ifb` for ingress) and falls
back to none with a warning if one is missing; runs `netem.sh up` + `apply` (robot i of the roster = `ss-r<i+1>`); starts
`match.launch.py netns:=true game_control_ip:=10.77.0.1` with `communication.discovery_address=10.77.0.255` so the broadcast
leaves through the veth; every brain gets the launch prefix `bash scripts/ns_exec.sh ss-rN` (sudo into the namespace, `env -i` with the
caller's whole environment so pixi/ROS/LD_LIBRARY_PATH/ROS_DOMAIN_ID survive, `setpriv` back to the user); and runs
`netem.sh down` in a `finally` plus `atexit` (Ctrl-C, stop button, normal exit; not on SIGKILL). The match bridge stays in the root
namespace and counts packets arriving at br-sim, i.e. after impairment.

## Per-receiver loss (`SS_NETEM_INGRESS=1` or the "Per-receiver loss" setting)

Egress netem loses a broadcast once, for everybody. With ingress mode each namespace gets an `ifb` device `ssi<N>`; `ssv<N>`
ingress has a `u32` filter on the team ports with `action mirred egress redirect dev ssi<N>`, and the profile (prio + netem) sits on
`ssi<N>`. The sender's egress is left clean (blackout still cuts it too), so robot N's profile describes N's *receiving* link.
Needs `modprobe ifb` (present in this WSL2 kernel).

Measured (rcap_bad on all, robot 1 broadcasts 1000 probes to 10.77.0.255:10131, receivers in ss-r2..4):

| mode | loss r2 / r3 / r4 | lost by r2 and r3 |
|---|---|---|
| egress (default) | 14.7% / 14.7% / 14.7%, identical sets | all of them |
| ingress | 16.7% / 14.7% / 13.8% (167/147/138 pkts) | 28 (independent expectation ~25); all three: 1 |
