"""scripts/ros_env.sh: sourcing a colcon overlay built in one pixi env must not drag python (and onnxruntime) back to
that env when the match runs in another (`pixi run -e gpu`). Fake envs and a fake overlay, plain bash, no ROS."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROS_ENV = Path(__file__).resolve().parents[1] / "scripts" / "ros_env.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _fake_envs(tmp_path: Path) -> tuple[Path, Path, Path]:
    default, gpu = tmp_path / ".pixi" / "envs" / "default", tmp_path / ".pixi" / "envs" / "gpu"
    for env in (default, gpu):
        (env / "bin").mkdir(parents=True)
        (env / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    # What the default env's ROS local_setup.bash does: its bin/ and site-packages go in front.
    (default / "local_setup.bash").write_text(
        f'export PATH="{default}/bin:$PATH"\n'
        f'export PYTHONPATH="{default}/lib/python3.12/site-packages${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        f'export AMENT_PREFIX_PATH="{default}${{AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}}"\n')
    ws = tmp_path / "ros_ws"
    (ws / "install" / "brain" / "lib").mkdir(parents=True)
    # Shape of colcon's prefix_chain.bash: the build env is baked in and chain-sourced first.
    (ws / "install" / "setup.bash").write_text(
        f'COLCON_CURRENT_PREFIX="{default}"\nsource "$COLCON_CURRENT_PREFIX/local_setup.bash"\n'
        f'COLCON_CURRENT_PREFIX="{ws}/install"\n'
        f'export PYTHONPATH="{ws}/install/brain/lib/python3.12/site-packages:$PYTHONPATH"\n'
        f'export LD_LIBRARY_PATH="{ws}/install/brain/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"\n')
    return default, gpu, ws


def _source(ws: Path, conda_prefix: Path, path: str) -> dict[str, str]:
    script = (f'set -euo pipefail; set +u; source "{ROS_ENV}"; ss_source_overlay "{ws}"; set -u\n'
              'for v in PATH PYTHONPATH AMENT_PREFIX_PATH LD_LIBRARY_PATH; do printf "%s=%s\\n" "$v" "${!v-}"; done')
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": path, "CONDA_PREFIX": str(conda_prefix), "PYTHONPATH": "/vendor"}
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


def test_overlay_from_default_env_keeps_the_active_gpu_env_in_front(tmp_path):
    default, gpu, ws = _fake_envs(tmp_path)
    got = _source(ws, gpu, f"{gpu}/bin:/usr/bin:/bin")
    path, pythonpath = got["PATH"].split(":"), got["PYTHONPATH"].split(":")
    assert path[0] == f"{gpu}/bin" and path.count(f"{gpu}/bin") == 1  # `python` resolves to the gpu env
    assert "/usr/bin" in path
    assert pythonpath == [f"{ws}/install/brain/lib/python3.12/site-packages", f"{gpu}/lib/python3.12/site-packages", "/vendor"]
    assert got["AMENT_PREFIX_PATH"] == str(gpu)
    assert got["LD_LIBRARY_PATH"] == f"{ws}/install/brain/lib"
    assert str(default) not in "".join(got.values())


def test_overlay_in_its_own_build_env_is_unchanged(tmp_path):
    default, _, ws = _fake_envs(tmp_path)
    got = _source(ws, default, f"{default}/bin:/usr/bin:/bin")
    assert got["PATH"].split(":")[0] == f"{default}/bin"
    assert got["PYTHONPATH"].split(":")[1] == f"{default}/lib/python3.12/site-packages"
