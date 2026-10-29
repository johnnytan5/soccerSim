"""One robot's command side, in its own process: the loco RPC service, /kick_ball and brain's text topics.

A brain sends ~300 loco RPCs a second. Handled in the sim process, ten of them compete with the physics loop
for the interpreter lock and the match runs below 50 Hz; here each robot's RPCs cost the sim nothing. The sim
and the gateways share one small memory-mapped file (like the viewer link): the gateway writes what brain
commanded, the sim writes the few things it decides (stand up).

    python -m soccer_sim_bridge.gateway STATE_FILE N_ROBOTS INDEX DOMAIN_ID NAME
"""

import os
import sys
import time

import numpy as np

# Row layout, one row of NF float64 per robot (the sim and every gateway use the same constants).
MODE, VX, VY, VYAW, HEAD_PITCH, HEAD_YAW, KICK_ACTIVE, KICK_VERSION, GETUP = range(9)
KICK_VALID, KICK_X, KICK_Y, KICK_DIR, KICK_POWER, KICK_STAMP, DECISION_TOPIC = range(9, 16)
STAND_UP, ALIVE, STOP = 16, 17, 18
WM_SEQ, WM_LEN, WM_STAMP, WM_SEEN = 19, 20, 21, 22  # world-model slot: seqlock counter, length, monotonic stamp, ever seen
NF = 24
TEXT_BYTES = 256
WM_BYTES = 8192  # one brain's /brain/world_model snapshot (about 2-3 KB of JSON for a 5v5 team)
DECISION, EVENT = 0, 1


class Shared:
    """The memory-mapped state: n float rows, then two text slots (decision, event) per robot, then a world-model slot
    per robot (a seqlock: the gateway writes it, the match process reads it, neither ever blocks)."""

    def __init__(self, path: str, n: int, create: bool = False):
        self.n = n
        self._float_bytes = n * NF * 8
        text_bytes = n * 2 * TEXT_BYTES
        size = self._float_bytes + text_bytes + n * WM_BYTES
        if create:
            with open(path, "wb") as f:
                f.truncate(size)
        self._mm = np.memmap(path, dtype=np.uint8, mode="r+", shape=(size,))
        self.rows = self._mm[: self._float_bytes].view(np.float64).reshape(n, NF)
        self._text = self._mm[self._float_bytes:self._float_bytes + text_bytes].reshape(n, 2, TEXT_BYTES)
        self._wm = self._mm[self._float_bytes + text_bytes:].reshape(n, WM_BYTES)

    def set_text(self, i: int, slot: int, text: str) -> None:
        raw = text.encode()[: TEXT_BYTES - 1]
        self._text[i, slot, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        self._text[i, slot, len(raw):] = 0

    def get_text(self, i: int, slot: int) -> str:
        return bytes(self._text[i, slot]).rstrip(b"\0").decode(errors="replace")

    def set_wm(self, i: int, data: bytes) -> bool:
        """Publish robot i's world-model snapshot. False (and the old one stays) when it does not fit."""
        if len(data) > WM_BYTES:
            return False
        row = self.rows[i]
        row[WM_SEQ] += 1.0  # odd: a write is in progress
        self._wm[i, : len(data)] = np.frombuffer(data, dtype=np.uint8)
        row[WM_LEN] = float(len(data))
        row[WM_STAMP] = time.monotonic()
        row[WM_SEEN] = 1.0
        row[WM_SEQ] += 1.0  # even: complete
        return True

    def get_wm(self, i: int, tries: int = 4) -> tuple[float, bytes] | None:
        """(stamp, snapshot) of robot i, or None when it never published or a consistent copy was not obtained."""
        row = self.rows[i]
        if not row[WM_SEEN]:
            return None
        for _ in range(tries):
            seq = row[WM_SEQ]
            if int(seq) % 2:
                continue
            data = bytes(self._wm[i, : int(row[WM_LEN])])
            stamp = float(row[WM_STAMP])
            if row[WM_SEQ] == seq:
                return stamp, data
        return None


def run(path: str, n: int, index: int, domain: int, name: str, parent_pid: int) -> None:
    # Imported here so `Shared` stays importable without ROS (tests, the sim process).
    import rclpy
    from booster_msgs.msg import RpcReqMsg, RpcRespMsg
    from brain.msg import Kick
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import String

    from soccer_sim.loco_rpc import MODE_DAMPING, MODE_PREPARE, MODE_SOCCER, LocoState, handle_request

    shared = Shared(path, n)
    row = shared.rows[index]
    loco = LocoState(mode=int(row[MODE]))
    ctx = rclpy.Context()
    rclpy.init(args=[], context=ctx, domain_id=domain)
    node = Node(name, context=ctx, start_parameter_services=False, enable_rosout=False)
    resp_pub = node.create_publisher(RpcRespMsg, "LocoApiTopicResp", 10)

    def push() -> None:
        row[MODE], row[VX], row[VY], row[VYAW] = loco.mode, loco.vx, loco.vy, loco.vyaw
        row[HEAD_PITCH], row[HEAD_YAW] = loco.head_pitch, loco.head_yaw
        row[KICK_ACTIVE], row[KICK_VERSION] = float(loco.visual_kick_active), loco.visual_kick_version
        if loco.getup_requested:
            row[GETUP] += 1.0  # a counter: the sim acts on every change, so a request is never lost
            loco.getup_requested = False

    def poll() -> None:
        if row[STAND_UP]:
            row[STAND_UP] = 0.0
            if loco.mode in (MODE_DAMPING, MODE_PREPARE):
                loco.mode = MODE_SOCCER
                push()

    def on_rpc(msg) -> None:
        poll()
        reply = handle_request(loco, msg.header, msg.body)
        resp_pub.publish(RpcRespMsg(uuid=msg.uuid, header=reply.header, body=reply.body))
        push()

    def on_kick(msg) -> None:
        row[KICK_X], row[KICK_Y], row[KICK_DIR], row[KICK_POWER] = float(msg.x), float(msg.y), float(msg.dir), float(msg.power)
        row[KICK_STAMP] = time.monotonic()
        row[KICK_VALID] = 1.0

    def on_speak(msg) -> None:
        # Sentences are events; single words are decisions, which only count until /brain/decision shows up.
        if " " in msg.data.strip():
            shared.set_text(index, EVENT, msg.data)
        elif not row[DECISION_TOPIC]:
            shared.set_text(index, DECISION, msg.data)

    def on_decision(msg) -> None:
        row[DECISION_TOPIC] = 1.0
        shared.set_text(index, DECISION, msg.data or "-")

    def on_world_model(msg) -> None:
        shared.set_wm(index, msg.data.encode())

    node.create_subscription(String, "/brain/world_model", on_world_model, 10)
    node.create_subscription(RpcReqMsg, "LocoApiTopicReq", on_rpc, 10)
    node.create_subscription(Kick, "/kick_ball", on_kick, 10)
    node.create_subscription(String, "/speak", on_speak, 10)
    node.create_subscription(String, "/brain/decision", on_decision, 10)
    node.create_timer(0.02, poll)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(node)
    push()
    try:
        while not shared.rows[0][STOP] and os.getppid() == parent_pid:
            row[ALIVE] = time.monotonic()
            executor.spin_once(timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        ctx.try_shutdown()


if __name__ == "__main__":
    path, n, index, domain, name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    run(path, n, index, domain, name, os.getppid())
