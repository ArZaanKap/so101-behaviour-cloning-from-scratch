"""Launch an Isaac Lab env from our own scripts (same CLI as ``isaaclab zero_agent``).

Usage in a script::

    from so1.sim import make_env, parse_cli

    args = parse_cli("what this script does", extra_args_fn)
    with make_env(args) as env:
        obs, _ = env.reset()
        ...

``parse_cli`` accepts the usual Isaac Lab flags (``--task``, ``--num_envs``, ``--viz kit|none``,
``--device``) plus Hydra-style preset overrides such as ``physics=isaacsim_physx``.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Callable, Iterator

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import validate

import isaaclab_tasks  # noqa: F401  -- registers Isaac Lab's own tasks
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import so1  # noqa: F401  -- registers the SO1-* tasks

DEFAULT_TASK = "SO1-Stack-Cube-SO101-IK-Abs-v0"


def parse_cli(
    description: str, add_args: Callable[[argparse.ArgumentParser], None] | None = None
) -> argparse.Namespace:
    """Parse Isaac Lab launcher flags + script-specific flags; leave preset overrides for Hydra."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=1)
    if add_args is not None:
        add_args(parser)
    add_launcher_args(parser)
    # Task cfg picks the device; default to no viewer (pass --viz kit to watch).
    parser.set_defaults(device=None, visualizer=[])
    args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + hydra_args  # Hydra reads preset overrides (physics=...) from argv
    return args


@contextlib.contextmanager
def make_env(args: argparse.Namespace) -> Iterator[gym.Env]:
    """Build the env cfg, launch the simulator, and yield the gym env (closed on exit)."""
    env_cfg, _ = resolve_task_config(args.task, "")
    apply_env_overrides(args, env_cfg)
    args.device = env_cfg.sim.device
    validate(env_cfg)
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        env = gym.make(args.task, cfg=env_cfg)
        cleanup.callback(lambda: close_env(env))
        yield env
