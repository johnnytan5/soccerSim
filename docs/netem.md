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

Gilbert-Elliott: P(good->bad)=5%, P(bad->good)=20% (mean burst ~5 packets), 75% loss in the bad state, 0.1% in good;
stationary loss 0.2*0.75+0.8*0.001 = 15.1%. Optionally add `rate 2mbit` to the rcap_bad line in `profile_args`.

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
- The blackout revert timer is a background shell job: it does not survive `down` or a reboot.
- Not integrated with `soccer_launcher` yet.

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
