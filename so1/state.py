"""Low-dimensional state of the SO-101 stack task, in the ROBOT BASE frame.

One function, ``get_state(env)``, used by BOTH the demo recorder and the policy rollout, so the
policy sees exactly the same observation layout at train and test time.

Why our own instead of the task's ``policy`` obs group: those terms mix frames (``eef_pos`` is
env-local world, ``cube_positions`` is raw world), while the IK action is in the robot base frame.
Here everything is in the base frame, same as the actions.
"""

from __future__ import annotations

import torch

# Tool-center-point (between the fingertips) in the gripper-link frame. From the URDF
# ``gripper_frame_joint`` origin; the sim USD keeps the same gripper link frame (checked with
# scripts/probe_ee.py: fingers point along the gripper link's -z).
TCP_OFFSET_IN_GRIPPER = (-0.0079, -0.000218, -0.0981274)

# Order + sizes of the keys returned by get_state (also the HDF5 obs layout).
STATE_KEYS: dict[str, int] = {
    "joint_pos": 6,  # 5 arm joints + gripper [rad]
    "joint_vel": 6,  # [rad/s]
    "eef_pos": 3,  # gripper link origin (= what the IK action targets) [m]
    "eef_quat": 4,  # gripper link orientation, xyzw
    "tcp_pos": 3,  # point between the fingertips [m]
    "cube_1_pos": 3,  # blue (bottom of the stack)
    "cube_1_quat": 4,
    "cube_2_pos": 3,  # red (the one to pick up)
    "cube_2_quat": 4,
    "cube_3_pos": 3,  # green (distractor)
    "cube_3_quat": 4,
}


def get_state(env) -> dict[str, torch.Tensor]:
    """Return ``{key: (num_envs, dim) float32 tensor}`` for every key in ``STATE_KEYS``."""
    import isaaclab.utils.math as math_utils

    uenv = env.unwrapped
    scene = uenv.scene
    robot = scene["robot"]
    root_pos = robot.data.root_pos_w.torch
    root_quat = robot.data.root_quat_w.torch

    def to_base(pos_w, quat_w):
        return math_utils.subtract_frame_transforms(root_pos, root_quat, pos_w, quat_w)

    grip_id = robot.body_names.index("gripper")
    eef_pos, eef_quat = to_base(robot.data.body_pos_w.torch[:, grip_id], robot.data.body_quat_w.torch[:, grip_id])
    tcp_off = torch.tensor(TCP_OFFSET_IN_GRIPPER, device=eef_pos.device).expand_as(eef_pos)
    tcp_pos = eef_pos + math_utils.quat_apply(eef_quat, tcp_off)

    state = {
        "joint_pos": robot.data.joint_pos.torch,
        "joint_vel": robot.data.joint_vel.torch,
        "eef_pos": eef_pos,
        "eef_quat": eef_quat,
        "tcp_pos": tcp_pos,
    }
    for name in ["cube_1", "cube_2", "cube_3"]:
        cube = scene[name]
        state[f"{name}_pos"], state[f"{name}_quat"] = to_base(cube.data.root_pos_w.torch, cube.data.root_quat_w.torch)

    return {k: state[k].float().clone() for k in STATE_KEYS}
