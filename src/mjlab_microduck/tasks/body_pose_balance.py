from pathlib import Path

import numpy as np
import torch
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.recorder_manager import (
    RecorderTerm,
    RecorderTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp


def apply_gui_balance_kick(
    env,
    env_ids,
):
    """
    Consume a pending GUI kick and apply it once.

    The kick is defined in WORLD frame:
        +x: world +x
        +y: world +y
    """
    del env_ids  # step events receive env_ids=None

    pending = getattr(
        env,
        "_pending_balance_kick",
        None,
    )

    if pending is None:
        return

    # Consume immediately: one click -> one kick.
    env._pending_balance_kick = None

    env_idx, delta_vx, delta_vy = pending

    ids = torch.tensor(
        [env_idx],
        dtype=torch.long,
        device=env.device,
    )

    push_by_setting_velocity(
        env,
        ids,
        velocity_range={
            "x": (delta_vx, delta_vx),
            "y": (delta_vy, delta_vy),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    )


def lateral_velocity_kick(
    env,
    env_ids,
    kick_time_s: float,
    delta_vy: float,
    asset_cfg: SceneEntityCfg,
):
    del env_ids

    if not hasattr(env, "_lateral_kick_dvy"):
        env._lateral_kick_dvy = torch.zeros(env.num_envs, device=env.device)

    env._lateral_kick_dvy.zero_()

    target_step = round(kick_time_s / env.step_dt)
    ids = torch.nonzero(env.episode_length_buf == target_step, as_tuple=False).squeeze(
        -1
    )
    if ids.numel() == 0:
        return

    velocity_mdp.push_by_setting_velocity(
        env,
        ids,
        velocity_range={
            "x": (0.0, 0.0),
            "y": (delta_vy, delta_vy),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        asset_cfg=asset_cfg,
    )
    env._lateral_kick_dvy[ids] = delta_vy


def quat_to_roll_pitch(q):
    """
    q: (..., 4), MuJoCo/mjlab quaternion order = [w, x, y, z]
    """
    w, x, y, z = q.unbind(-1)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    return roll, pitch


class BalanceTraceRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.env_idx = cfg.params.get("env_idx", 0)
        self.path = Path(cfg.params["path"])

        self.robot = env.scene["robot"]
        self.contact_sensor = env.scene.sensors["feet_ground_contact"]

        site_ids, _ = self.robot.find_sites(
            ("left_foot", "right_foot"),
            preserve_order=True,
        )

        self.left_foot_id = site_ids[0]
        self.right_foot_id = site_ids[1]
        self.root_body_id = self.robot.indexing.root_body_id

        self.rows = []
        self.columns = [
            "time",
            "com_x",
            "com_y",
            "com_z",
            "left_foot_x",
            "left_foot_y",
            "left_foot_z",
            "right_foot_x",
            "right_foot_y",
            "right_foot_z",
            "contact_left",
            "contact_right",
            "roll",
            "pitch",
            "base_vx_w",
            "base_vy_w",
            "base_vz_w",
            "base_vx_b",
            "base_vy_b",
            "base_vz_b",
            "base_wx_b",
            "base_wy_b",
            "base_wz_b",
            "kick_dvy",
            "terminal",
        ]

    def _append(self, terminal: float = 0.0):
        e = self.env_idx
        env = self._env

        # whole-robot CoM
        com_w = env.sim.data.subtree_com[e, self.root_body_id]

        # feet
        left_foot_w = self.robot.data.site_pos_w[e, self.left_foot_id]
        right_foot_w = self.robot.data.site_pos_w[e, self.right_foot_id]

        # contacts
        found = self.contact_sensor.data.found[e, :2]
        found = (
            found.reshape(2, -1).any(dim=-1).float()
        )  # .reshape(2, -1) is for multiple sub founds.

        # orientation
        quat_w = self.robot.data.root_link_quat_w[e]
        roll, pitch = quat_to_roll_pitch(quat_w)

        # velocities
        v_w = self.robot.data.root_link_lin_vel_w[e]
        v_b = self.robot.data.root_link_lin_vel_b[e]
        omega_b = self.robot.data.root_link_ang_vel_b[e]

        # perturbation marker
        if hasattr(env, "_lateral_kick_dvy"):
            kick = env._lateral_kick_dvy[e]
        else:
            kick = torch.zeros((), device=env.device)

        t = env.episode_length_buf[e].float() * env.step_dt

        row = torch.cat(
            [
                t.view(1),
                com_w,
                left_foot_w,
                right_foot_w,
                found,
                torch.stack([roll, pitch]),
                v_w,
                v_b,
                omega_b,
                kick.view(1),
                torch.tensor(
                    [terminal],
                    device=env.device,
                    dtype=torch.float32,
                ),
            ]
        )

        self.rows.append(row.detach().cpu().numpy())

    def record_post_step(self):
        # 如果刚刚 reset，不把新 episode 的 initial state
        # 当成上一 episode 的 continuation
        if not bool(self._env.reset_buf[self.env_idx].item()):
            self._append(terminal=0.0)

    def record_pre_reset(self, env_ids):
        # 在摔倒/reset 前保留最后状态
        if torch.any(env_ids == self.env_idx):
            self._append(terminal=1.0)

    def close(self):
        if len(self.rows) == 0:
            return

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            self.path,
            data=np.stack(self.rows),
            columns=np.array(self.columns),
        )
