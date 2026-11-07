"""SO-101 cube stacking, copied from Isaac Lab v3.0.0 (contrib/stack/config/so101) so we own it.

Registers OUR task IDs (prefixed ``SO1-``) so they never collide with Isaac Lab's originals.
"""

import gymnasium as gym

# Absolute end-effector pose control (IK). 8D action:
# [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper_closedness]
gym.register(
    id="SO1-Stack-Cube-SO101-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_abs_env_cfg:SO101CubeStackEnvCfg",
    },
    disable_env_checker=True,
)
