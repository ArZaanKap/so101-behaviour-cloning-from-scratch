"""Probe the SO-101 task: print the frames/numbers the scripted expert needs.

Run from C:\\IsaacLabProjects\\SO1:

    uv run --project C:\\IsaacLab3 --extra isaacsim python -m scripts.probe_ee physics=isaacsim_physx

Prints (all in the ROBOT BASE frame, which is the frame the IK action uses):
  1. robot body names, base pose in world, env origin
  2. gripper-link pose at reset + its x/y/z axes
  3. moving-jaw origin and the URDF tool-center-point (TCP) estimate
  4. cube positions
Then commands "keep position, point straight down" for 2 s and reports what the IK achieved.
"""

import torch

from so1.sim import make_env, parse_cli

# URDF ``gripper_frame_joint`` origin: TCP (between the fingertips) expressed in the gripper link frame.
TCP_OFFSET_IN_GRIPPER = (-0.0079, -0.000218, -0.0981274)


def fmt(t: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in t.tolist()) + "]"


def main():
    import isaaclab.utils.math as math_utils

    args = parse_cli("Probe SO-101 frames for the scripted expert.")
    with make_env(args) as env:
        env.reset()
        uenv = env.unwrapped
        robot = uenv.scene["robot"]
        device = uenv.device
        num_envs = uenv.num_envs
        grip_id = robot.body_names.index("gripper")
        jaw_id = robot.body_names.index("moving_jaw_so101_v1")

        def base_pose():
            return robot.data.root_pos_w.torch, robot.data.root_quat_w.torch

        def to_base(pos_w, quat_w=None):
            rp, rq = base_pose()
            return math_utils.subtract_frame_transforms(rp, rq, pos_w, quat_w)

        def gripper_pose_b():
            return to_base(robot.data.body_pos_w.torch[:, grip_id], robot.data.body_quat_w.torch[:, grip_id])

        # Hold still for 1 s (20 policy steps) so everything settles after reset.
        pos_b, quat_b = gripper_pose_b()
        hold = torch.cat([pos_b, quat_b, torch.zeros(num_envs, 1, device=device)], dim=1)  # c=0: open
        for _ in range(20):
            env.step(hold)

        print("\n================ PROBE: static info (env 0) ================")
        print("body names       :", robot.body_names)
        print("joint names      :", robot.joint_names)
        rp, rq = base_pose()
        print("base pos (world) :", fmt(rp[0]), " quat xyzw:", fmt(rq[0]))
        print("env origin       :", fmt(uenv.scene.env_origins[0]))
        print("joint pos [rad]  :", fmt(robot.data.joint_pos.torch[0]))

        pos_b, quat_b = gripper_pose_b()
        R = math_utils.matrix_from_quat(quat_b)  # columns = gripper axes expressed in base frame
        print("\n--- gripper link (IK target body) in BASE frame ---")
        print("pos              :", fmt(pos_b[0]))
        print("quat xyzw        :", fmt(quat_b[0]))
        print("x axis           :", fmt(R[0, :, 0]))
        print("y axis           :", fmt(R[0, :, 1]))
        print("z axis           :", fmt(R[0, :, 2]), " (fingers point along -z if URDF frames hold)")

        jaw_b, _ = to_base(robot.data.body_pos_w.torch[:, jaw_id], robot.data.body_quat_w.torch[:, jaw_id])
        tcp_off = torch.tensor(TCP_OFFSET_IN_GRIPPER, device=device).expand(num_envs, 3)
        tcp_b = pos_b + math_utils.quat_apply(quat_b, tcp_off)
        print("moving jaw origin:", fmt(jaw_b[0]))
        print("TCP (URDF offset):", fmt(tcp_b[0]), " = gripper pos + R @", TCP_OFFSET_IN_GRIPPER)

        print("\n--- cubes in BASE frame (pos) ---")
        for name in ["cube_1", "cube_2", "cube_3"]:
            c = uenv.scene[name]
            cp, cq = to_base(c.data.root_pos_w.torch, c.data.root_quat_w.torch)
            print(f"{name:7s}          :", fmt(cp[0]), " quat xyzw:", fmt(cq[0]))

        # ---- Top-down test: keep gripper position, rotate so gripper z = base +z (fingers down) ----
        z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_envs, 3)
        x_proj = R[:, :, 0].clone()
        x_proj[:, 2] = 0.0
        x = x_proj / x_proj.norm(dim=1, keepdim=True)
        y = torch.cross(z, x, dim=1)
        R_des = torch.stack([x, y, z], dim=2)
        quat_des = math_utils.quat_from_matrix(R_des)
        target = torch.cat([pos_b, quat_des, torch.zeros(num_envs, 1, device=device)], dim=1)
        for _ in range(40):  # 2 s
            env.step(target)
        pos2, quat2 = gripper_pose_b()
        R2 = math_utils.matrix_from_quat(quat2)
        tilt_deg = torch.rad2deg(torch.acos(R2[:, 2, 2].clamp(-1, 1)))
        print("\n--- top-down test (2 s): keep pos, gripper z -> base +z ---")
        print("pos error [m]    :", f"{(pos2 - pos_b).norm(dim=1)[0].item():.4f}")
        print("z-axis tilt [deg]:", f"{tilt_deg[0].item():.2f}  (0 = perfectly top-down)")
        print("joint pos [rad]  :", fmt(robot.data.joint_pos.torch[0]))
        print("gripper joint    :", f"{robot.data.joint_pos.torch[0, -1].item():.4f}  (open target 1.745)")
        print("==============================================================\n")


if __name__ == "__main__":
    main()
