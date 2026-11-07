"""Scripted expert for the SO-101 single-stack task (red cube_2 onto blue cube_1).

It uses PRIVILEGED state (true cube poses from the sim) and runs a fixed sequence of phases:

    APPROACH -> DESCEND -> CLOSE -> LIFT -> MOVE -> LOWER -> RELEASE -> RETREAT -> DONE

Everything is in the robot BASE frame (see so1/state.py). Design choices:
- We plan for the TCP (point between the fingertips). The IK action controls the TCP directly
  (``body_offset`` in the task cfg), so the action xyz IS the TCP target.
- Top-down grasps only: gripper z axis = base +z, so the commanded orientation is a pure yaw
  rotation. The yaw matches the cube's faces (mod 90 deg) and is picked closest to the arm's
  natural heading, so wrist_roll never has to turn more than 45 deg.
- The commanded TCP moves at most ``MAX_STEP`` per action (the IK takes one linearized step per
  physics tick, so big target jumps would give jerky motion and jerky demos).
- Batched over envs: every quantity is a (num_envs, ...) tensor, and each env has its own phase.

Output action (num_envs, 8): [tcp_x, tcp_y, tcp_z, qx, qy, qz, qw, gripper_closedness].
"""

from __future__ import annotations

import math

import torch

# ---- phases ----
APPROACH, DESCEND, CLOSE, LIFT, MOVE, LOWER, RELEASE, RETREAT, DONE = range(9)
PHASE_NAMES = ["APPROACH", "DESCEND", "CLOSE", "LIFT", "MOVE", "LOWER", "RELEASE", "RETREAT", "DONE"]

# ---- tunables (meters / policy steps at 20 Hz) ----
CUBE_SIZE = 0.0406  # cube edge (cube center rests 0.0203 above the table)
HOVER = 0.05  # TCP height above the cube center while approaching
GRASP_DZ = 0.0  # TCP height relative to cube center when closing
# The URDF "TCP" sits where the CLOSED jaws meet, i.e. at the fixed jaw. The moving jaw hinges on
# the gripper's +x side, so the open gap is at +x. Put the cube center this far along gripper +x from
# the TCP (~ half the cube + a few mm of clearance from the fixed jaw).
GRASP_DX = 0.024
CARRY_CLEARANCE = 0.05  # held cube bottom clearance above cube_1 top while moving
PLACE_GAP = 0.008  # held cube bottom above cube_1 top when we let go (it drops this much)
RETREAT_DZ = 0.06
MAX_STEP = 0.01  # max commanded TCP motion per action [m] (= 0.2 m/s)
TOL_COARSE = 0.012  # "reached" tolerance for free-space moves [m]
TOL_FINE_Z = 0.006  # height tolerance before closing / releasing [m]
TOL_FINE_XY = 0.012  # sideways tolerance before closing / releasing [m]. At the far edge of the
# workspace the 5-DOF IK leaves ~7 mm of lateral error; stacking only needs < 4 cm.
CLOSE_STEPS = 20  # 1.0 s to close the (deliberately slow) gripper
RELEASE_STEPS = 25  # 1.25 s to open fully (success needs the jaw fully open)
PHASE_TIMEOUT = 150  # 7.5 s stuck in one phase -> mark this attempt as failed

GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 1.0  # action closedness


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw of an xyzw quaternion (assumes the body is upright, e.g. a cube resting flat)."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion of a pure rotation about +z (= top-down gripper with that yaw)."""
    zeros = torch.zeros_like(yaw)
    return torch.stack([zeros, zeros, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)


class ScriptedExpert:
    def __init__(self, num_envs: int, device: str | torch.device):
        self.num_envs = num_envs
        self.device = device
        n = num_envs
        self.phase = torch.full((n,), APPROACH, dtype=torch.long, device=device)
        self.phase_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.cmd_tcp = torch.zeros(n, 3, device=device)  # rate-limited commanded TCP position
        self.yaw = torch.zeros(n, device=device)  # commanded gripper yaw
        self.grasp_off = torch.zeros(n, 3, device=device)  # cube_2 center relative to TCP, set at grasp
        self.hold_wp = torch.zeros(n, 3, device=device)  # waypoint frozen at phase entry (RELEASE/RETREAT)

    # ------------------------------------------------------------------ public API
    def reset(self, env_ids: torch.Tensor, state: dict[str, torch.Tensor]) -> None:
        """Start a new attempt for ``env_ids`` (call right after the env reset them)."""
        if len(env_ids) == 0:
            return
        self.phase[env_ids] = APPROACH
        self.phase_steps[env_ids] = 0
        self.failed[env_ids] = False
        self.cmd_tcp[env_ids] = state["tcp_pos"][env_ids]
        self.grasp_off[env_ids] = 0.0
        self.yaw[env_ids] = self._grasp_yaw(state["cube_2_pos"][env_ids], state["cube_2_quat"][env_ids])

    def act(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Advance each env's phase machine one step and return the (num_envs, 8) action."""
        self._update_phases(state)
        waypoint = self._waypoint(state)

        # Rate-limit the commanded TCP toward the waypoint.
        delta = waypoint - self.cmd_tcp
        dist = delta.norm(dim=1, keepdim=True).clamp(min=1e-9)
        self.cmd_tcp = self.cmd_tcp + delta * (MAX_STEP / dist).clamp(max=1.0)

        quat = _quat_from_yaw(self.yaw)
        closed = (self.phase >= CLOSE) & (self.phase <= LOWER)
        grip = torch.where(closed, GRIPPER_CLOSED, GRIPPER_OPEN).unsqueeze(1).to(self.cmd_tcp.dtype)
        return torch.cat([self.cmd_tcp, quat, grip], dim=1)

    # ------------------------------------------------------------------ internals
    def _grasp_yaw(self, cube_pos: torch.Tensor, cube_quat: torch.Tensor) -> torch.Tensor:
        """Cube-face yaw (mod 90 deg) closest to the arm's natural heading toward the cube."""
        natural = torch.atan2(cube_pos[:, 1], cube_pos[:, 0])  # gripper x points radially outward
        d = _wrap(_yaw_from_quat(cube_quat) - natural)
        d = (d + math.pi / 4) % (math.pi / 2) - math.pi / 4  # nearest equivalent face, in [-45, 45) deg
        return _wrap(natural + d)

    def _waypoint(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Target TCP position for each env's current phase."""
        c1, c2 = state["cube_1_pos"], state["cube_2_pos"]
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        # Where the TCP must be for the HELD cube's center to sit at a given point.
        place_center = c1 + up * CUBE_SIZE
        carry_center = c1 + up * (CUBE_SIZE + CARRY_CLEARANCE)

        # TCP target that puts the cube center inside the open jaw gap (GRASP_DX along gripper +x).
        shift = torch.stack([torch.cos(self.yaw), torch.sin(self.yaw), torch.zeros_like(self.yaw)], dim=1) * GRASP_DX
        grasp_tcp = c2 - shift

        wp = torch.empty_like(c2)
        p = self.phase
        wp[:] = self.cmd_tcp  # default: hold
        m = p == APPROACH
        wp[m] = grasp_tcp[m] + up * HOVER
        m = p == DESCEND
        wp[m] = grasp_tcp[m] + up * GRASP_DZ
        m = p == CLOSE
        wp[m] = grasp_tcp[m] + up * GRASP_DZ
        m = p == LIFT
        wp[m] = torch.cat([self.cmd_tcp[m, :2], (carry_center[m, 2:] - self.grasp_off[m, 2:])], dim=1)
        m = p == MOVE
        wp[m] = carry_center[m] - self.grasp_off[m]
        m = p == LOWER
        wp[m] = place_center[m] + up * PLACE_GAP - self.grasp_off[m]
        m = (p == RELEASE) | (p == RETREAT) | (p == DONE)
        wp[m] = self.hold_wp[m]
        return wp

    def _update_phases(self, state: dict[str, torch.Tensor]) -> None:
        tcp = state["tcp_pos"]
        wp = self._waypoint(state)
        err = (wp - tcp).norm(dim=1)
        fine = ((wp - tcp)[:, 2].abs() < TOL_FINE_Z) & ((wp - tcp)[:, :2].norm(dim=1) < TOL_FINE_XY)
        self.phase_steps += 1
        p = self.phase
        nxt = p.clone()

        nxt[(p == APPROACH) & (err < TOL_COARSE)] = DESCEND
        nxt[(p == DESCEND) & fine] = CLOSE
        done_closing = (p == CLOSE) & (self.phase_steps >= CLOSE_STEPS)
        nxt[done_closing] = LIFT
        nxt[(p == LIFT) & (err < TOL_COARSE)] = MOVE
        nxt[(p == MOVE) & (err < TOL_COARSE)] = LOWER
        to_release = (p == LOWER) & fine
        nxt[to_release] = RELEASE
        done_release = (p == RELEASE) & (self.phase_steps >= RELEASE_STEPS)
        nxt[done_release] = RETREAT
        nxt[(p == RETREAT) & (err < TOL_COARSE)] = DONE

        # Bookkeeping on phase entry.
        self.grasp_off[done_closing] = (state["cube_2_pos"] - tcp)[done_closing]  # remember the grip
        self.hold_wp[to_release] = tcp[to_release]
        self.hold_wp[done_release] = tcp[done_release] + torch.tensor([0.0, 0.0, RETREAT_DZ], device=self.device)

        changed = nxt != p
        self.phase_steps[changed] = 0
        self.phase = nxt
        self.failed |= (self.phase != DONE) & (self.phase_steps > PHASE_TIMEOUT)
