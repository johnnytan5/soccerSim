"""ffmpeg helpers shared by the replay renderer (scripts/render_replay.py) and the MuJoCo clip recorder.

NVENC (h264_nvenc, constant quality 28, preset p5) when the build and the GPU have it, libx264 otherwise.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_ffmpeg() -> str:
    for env in ("default", "gpu"):
        cand = ROOT / ".pixi" / "envs" / env / "bin" / "ffmpeg"
        if cand.exists():
            return str(cand)
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg not found (pixi env or PATH)")
    return found


def nvenc_ok(ffmpeg: str) -> bool:
    p = subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2", "-c:v", "h264_nvenc",
                        "-f", "null", "-"], capture_output=True)
    return p.returncode == 0


def encoder_args(ffmpeg: str, force_cpu: bool = False) -> tuple[list[str], str]:
    """(output codec arguments, name)."""
    if not force_cpu and nvenc_ok(ffmpeg):
        return (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "28", "-b:v", "0", "-maxrate", "2M", "-bufsize", "4M",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart"], "h264_nvenc")
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "28", "-maxrate", "2M", "-bufsize", "4M", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"], "libx264"


def open_encoder(ffmpeg: str, out: str, codec_args: list[str], w: int, h: int, fps: int) -> subprocess.Popen:
    """ffmpeg reading raw bgr24 frames on stdin."""
    return subprocess.Popen([ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
                             "-i", "-", "-an", *codec_args, out], stdin=subprocess.PIPE)
