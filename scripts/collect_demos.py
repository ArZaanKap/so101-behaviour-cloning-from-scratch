"""Run the scripted expert and save SUCCESSFUL episodes to HDF5.

Run from C:\\IsaacLabProjects\\SO1:

    # watch a few attempts (1 env, viewer on)
    uv run --project C:\\IsaacLab3 --extra isaacsim python -m scripts.collect_demos ^
        --num_envs 1 --num_demos 3 --viz kit --out data/debug.hdf5 physics=isaacsim_physx

    # collect a dataset (headless, parallel envs)
    uv run --project C:\\IsaacLab3 --extra isaacsim python -m scripts.collect_demos ^
        --num_envs 16 --num_demos 200 --out data/demos.hdf5 physics=isaacsim_physx

HDF5 layout (robomimic-style)::

    data/                       attrs: task, state_keys (json), action_layout, num_demos, total_samples
      demo_0/                   attrs: num_samples
        obs/<key>   (T, dim)    float32, keys + dims = so1.state.STATE_KEYS, base frame
        actions     (T, 8)      float32, [tcp_xyz, gripper_quat_xyzw, gripper_closedness]
        expert_phase (T,)       int8, which expert phase produced each step (debug/analysis)
      demo_1/ ...

Step t pairs the observation BEFORE the action with the action taken (obs_t -> action_t).
"""

import json
import os
import time

import torch

from so1.sim import make_env, parse_cli

ACTION_LAYOUT = "tcp_x, tcp_y, tcp_z, qx, qy, qz, qw, gripper_closedness(0=open,1=closed)"


def add_args(parser):
    parser.add_argument("--num_demos", type=int, default=100, help="Successful episodes to save.")
    parser.add_argument("--out", type=str, default="data/demos.hdf5")
    parser.add_argument("--max_attempts", type=int, default=None, help="Stop after this many attempts.")
    parser.add_argument("--debug", action="store_true", help="Print env 0's expert state every 10 steps.")


def main():
    import h5py

    from so1.expert import DONE, PHASE_NAMES, ScriptedExpert
    from so1.state import STATE_KEYS, get_state

    args = parse_cli("Collect scripted-expert demos for the SO-101 single-stack task.", add_args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    max_attempts = args.max_attempts or args.num_demos * 5

    with make_env(args) as env, h5py.File(args.out, "w") as f:
        uenv = env.unwrapped
        n, device = uenv.num_envs, uenv.device
        data = f.create_group("data")
        data.attrs["task"] = args.task
        data.attrs["state_keys"] = json.dumps(STATE_KEYS)
        data.attrs["action_layout"] = ACTION_LAYOUT

        expert = ScriptedExpert(n, device)
        env.reset()
        state = get_state(env)
        expert.reset(torch.arange(n, device=device), state)

        # Per-env episode buffers (lists of per-step CPU tensors).
        bufs = [{"obs": {k: [] for k in STATE_KEYS}, "actions": [], "phase": []} for _ in range(n)]
        saved, attempts, total_samples = 0, 0, 0
        fail_phase = [0] * len(PHASE_NAMES)  # where failed attempts ended
        t0 = time.time()

        step = 0
        while saved < args.num_demos and attempts < max_attempts and uenv.sim.is_running():
            action = expert.act(state)
            if args.debug and step % 10 == 0:
                f3 = lambda t: "[" + ", ".join(f"{v:+.3f}" for v in t.tolist()) + "]"  # noqa: E731
                wp = expert._waypoint(state)[0]
                print(
                    f"[debug] t={step:4d} {PHASE_NAMES[int(expert.phase[0])]:8s} tcp={f3(state['tcp_pos'][0])} "
                    f"wp={f3(wp)} err={(wp - state['tcp_pos'][0]).norm():.4f} cmd_tcp={f3(expert.cmd_tcp[0])} "
                    f"q={f3(state['joint_pos'][0])}",
                    flush=True,
                )
            step += 1
            state_cpu = {k: v.cpu() for k, v in state.items()}
            action_cpu, phase_cpu = action.cpu(), expert.phase.cpu()
            for i in range(n):
                for k in STATE_KEYS:
                    bufs[i]["obs"][k].append(state_cpu[k][i])
                bufs[i]["actions"].append(action_cpu[i])
                bufs[i]["phase"].append(phase_cpu[i])

            with torch.inference_mode():
                _, _, terminated, truncated, _ = env.step(action)
            success = uenv.termination_manager.get_term("success").clone()
            ended = terminated | truncated
            # Abort hopeless attempts early instead of waiting for the 30 s timeout: the expert is
            # stuck (phase timeout) or finished but the task never registered success.
            give_up = ~ended & (expert.failed | ((expert.phase == DONE) & (expert.phase_steps > 40)))
            aborted = give_up.nonzero().flatten()
            if len(aborted) > 0:
                uenv.reset(env_ids=aborted)
            done = (ended | give_up).nonzero().flatten()

            for i in done.tolist():
                attempts += 1
                ok = bool(ended[i]) and bool(success[i]) and not bool(expert.failed[i])
                if ok and saved < args.num_demos:
                    g = data.create_group(f"demo_{saved}")
                    T = len(bufs[i]["actions"])
                    g.attrs["num_samples"] = T
                    obs_g = g.create_group("obs")
                    for k in STATE_KEYS:
                        obs_g.create_dataset(k, data=torch.stack(bufs[i]["obs"][k]).numpy())
                    g.create_dataset("actions", data=torch.stack(bufs[i]["actions"]).numpy())
                    g.create_dataset("expert_phase", data=torch.stack(bufs[i]["phase"]).numpy().astype("int8"))
                    saved += 1
                    total_samples += T
                elif not ok:
                    fail_phase[int(expert.phase[i])] += 1
                bufs[i] = {"obs": {k: [] for k in STATE_KEYS}, "actions": [], "phase": []}
                rate = saved / attempts
                print(
                    f"[collect] attempt {attempts:4d} | saved {saved:4d}/{args.num_demos} | success {rate:5.1%} | "
                    f"last: {'OK  ' if ok else 'FAIL'} len={T if ok else '-'} | {time.time() - t0:6.0f}s",
                    flush=True,
                )

            # Envs that finished were auto-reset by Isaac Lab; restart the expert on their new state.
            state = get_state(env)
            expert.reset(done, state)

        data.attrs["num_demos"] = saved
        data.attrs["total_samples"] = total_samples
        print(f"\n[collect] saved {saved} demos ({total_samples} steps) to {os.path.abspath(args.out)}")
        print(f"[collect] attempts {attempts}, success rate {saved / max(attempts, 1):.1%}")
        fails = {PHASE_NAMES[p]: c for p, c in enumerate(fail_phase) if c}
        if fails:
            print(f"[collect] failed attempts ended in phase: {fails}")


if __name__ == "__main__":
    main()
