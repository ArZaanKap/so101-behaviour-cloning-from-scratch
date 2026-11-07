"""Smoke test: launch our SO-101 task and hold the arm still (zero-action agent).

Run from C:\\IsaacLabProjects\\SO1:

    uv run --project C:\\IsaacLab3 --extra isaacsim python -m scripts.check_env ^
        --task SO1-Stack-Cube-SO101-IK-Abs-v0 --num_envs 1 --visualizer kit physics=isaacsim_physx

Any flag Isaac Lab's ``isaaclab zero_agent`` accepts works here (e.g. ``--max_steps 200``,
``--headless``). For absolute IK actions, "zero" means "hold the current end-effector pose".
"""

import sys

# Warp captures ``enable_backward`` at module creation; set it before anything defines kernels
# (mirrors Isaac Lab's scripts/environments/zero_agent.py).
import warp as wp

wp.config.enable_backward = False

import so1  # noqa: E402, F401  -- registers the SO1-* task IDs with gymnasium
from isaaclab_rl.entrypoints.simple_agents import run  # noqa: E402

if __name__ == "__main__":
    run(sys.argv[1:], policy="zero")
