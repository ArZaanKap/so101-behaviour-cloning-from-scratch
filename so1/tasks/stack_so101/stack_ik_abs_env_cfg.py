# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab v3.0.0:
#   source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/so101/stack_ik_abs_env_cfg.py
# Changes vs upstream:
#   - Removed the XR-headset teleop pipeline (``_build_so101_stack_pipeline``, ``isaac_teleop``,
#     clutch home / orientation-offset constants). We collect demos with a scripted expert, not
#     XR teleop. The upstream file still has it if we ever need it.
#   - SO1 project change: SINGLE stack. Success = cube_2 (red) on cube_1 (blue); see the end of
#     ``__post_init__``.
#   - SO1 project change: the IK controls the TCP (``body_offset``), so the action's xyz is the
#     point between the fingertips, not the gripper-link origin.
#   Everything else (robot gains, IK controller settings, action layout) is unchanged.

from dataclasses import MISSING

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass, replace

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.utils import preset

from so1.state import TCP_OFFSET_IN_GRIPPER

from . import stack_joint_pos_env_cfg
from .pose_ik_action import SO101PoseIKActionCfg
from .pose_ik_controller import SO101PoseIKControllerCfg
from .stack_joint_pos_env_cfg import SO101_GRIPPER_CLOSE, SO101_GRIPPER_OPEN

##
# Pre-defined configs
##
from isaaclab_assets.robots.so101 import SO101_HIGH_PD_CFG  # isort: skip

# Analog-gripper action affine: ``joint = offset + scale * closedness`` maps a closedness
# ``c in [0, 1]`` to a ``gripper`` joint target [rad]. ``offset == OPEN`` (so ``c=0`` is fully
# open) and ``offset + scale == CLOSE`` (so ``c=1`` is fully closed).
_SO101_GRIPPER_OFFSET = SO101_GRIPPER_OPEN  # [rad] closedness c=0 -> open
_SO101_GRIPPER_SCALE = SO101_GRIPPER_CLOSE - SO101_GRIPPER_OPEN  # [rad] c=1 -> close


@configclass
class SO101IkActionsCfg:
    """Action terms for SO-101 IK-Abs, in field order ``[arm_action, gripper_action]``.

    The flattened 8D action is
    ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper_closedness]``:
    arm = 7 dims (absolute EE position + orientation, xyzw, in the robot base frame),
    gripper = 1 dim (0 = open, 1 = closed). A fresh (non-inheriting) container, because
    inheriting the base ``ActionsCfg`` would append fields after the base's and mis-order the concat.
    """

    arm_action: SO101PoseIKActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg = MISSING


@configclass
class SO101CubeStackEnvCfg(stack_joint_pos_env_cfg.SO101CubeStackEnvCfg):
    """SO-101 cube-stack environment with absolute task-space (IK) control.

    The SO-101 has only 5 actuated arm DOF, so a full 6-DOF pose target is over-determined by
    one DOF. The absolute end-effector SE3 pose is tracked by a full-pose differential IK over all
    5 arm joints: position rows exactly (weight 1), orientation rows soft-weighted
    (``orientation_weight``) so orientation is best-effort and never leaks error into position.
    Orientation is restricted to the wrist (``wrist_flex``, ``wrist_roll``) so ``shoulder_pan``
    serves position only.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Stiffer PD arm for IK tracking, plus a softened gripper: the asset default (10 N·m,
        # 10 rad/s) snaps shut and ejects the cube, so cap grip torque and closing speed.
        self.scene.robot = replace(
            self.scene.robot,
            soft_joint_pos_limit_factor=SO101_HIGH_PD_CFG.soft_joint_pos_limit_factor,
            spawn=replace(
                SO101_HIGH_PD_CFG.spawn,
                variants={
                    "Robot": "robot",
                    "Sensor": "sensors",
                    "Physics": preset(default="physx", isaacsim_physx="physx", physx="physx", newton_mjwarp="physics"),
                },
            ),
            actuators={
                "arm": SO101_HIGH_PD_CFG.actuators["arm"],
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper"],
                    joint_effort_limit=1.0,  # was 10.0 -- cap grip torque so it can't push through
                    joint_velocity_limit=2.0,  # was 10.0 -- close gently instead of snapping shut
                    stiffness=17.8,
                    damping=0.60,
                ),
            },
        )

        self.actions = SO101IkActionsCfg(
            # Full-pose SE3 IK over all 5 arm joints. Solves a 6-row pose task [pos_xyz, ori_xyz];
            # the 3 orientation rows are soft-weighted.
            arm_action=SO101PoseIKActionCfg(
                asset_name="robot",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body_name="gripper",
                # SO1 change: control the TCP (between the fingertips), not the gripper-link origin.
                # Top-down poses push wrist_flex near its limit, so the IK trades a few degrees of
                # tilt for position. With the link origin as the control point, that tilt swings
                # the fingertips ~1.6 cm off target. With the TCP as the control point, the residual
                # lands at the wrist. Action xyz = TCP position in the robot base frame.
                body_offset=SO101PoseIKActionCfg.OffsetCfg(pos=TCP_OFFSET_IN_GRIPPER),
                scale=1.0,
                controller=SO101PoseIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,  # ABSOLUTE targets (project decision 2026-11-07)
                    # Manipulability-aware damped least squares keeps the 5-DOF arm
                    # well-conditioned near singularities.
                    ik_method="adaptive_dls",
                    ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
                    orientation_weight=0.5,
                    # Orientation is a wrist-only task. Position still uses all 5 joints.
                    orientation_joint_names=("wrist_flex", "wrist_roll"),
                    # Null-space joint-limit avoidance (0 disables it).
                    joint_limit_avoidance_gain=0.5,
                    joint_limit_avoidance_margin=0.3,
                ),
            ),
            # Analog gripper: closedness c in [0, 1] -> joint target via ``offset + scale * c``.
            gripper_action=mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper"],
                scale=_SO101_GRIPPER_SCALE,
                offset=_SO101_GRIPPER_OFFSET,
                use_default_offset=False,
            ),
        )

        # ---------------- SO1 project changes ----------------
        # Single stack (decided 2026-11-07): success = cube_2 (red) on cube_1 (blue), gripper
        # released (fully open) and both cubes at rest. ``cube_3_cfg=None`` drops the green-on-red
        # check. cube_3 (green) stays in the scene as a distractor (still randomized and observed).
        self.terminations.success = DoneTerm(func=mdp.cubes_stacked, params={"cube_3_cfg": None})
