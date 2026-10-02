import copy

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommandCfg,
)

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

MicroduckStandingRlCfg = copy.deepcopy(MicroduckRlCfg)
MicroduckStandingRlCfg.experiment_name = "standing"
MicroduckStandingRlCfg.run_name = "standing"
MicroduckStandingRlCfg.logger = "tensorboard"
MicroduckStandingRlCfg.upload_model = False


def report_standing_height(env, env_ids):
    # Measure all environments; use num-envs=1 for the first measurement.
    del env_ids

    robot = env.scene["robot"]
    data = robot.data

    height = data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]

    # Angle between root-link local z-axis and world z-axis.
    quat = data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    tilt = torch.acos(cos_tilt.clamp(-1.0, 1.0))

    elapsed = env.episode_length_buf * env.step_dt

    linear_speed = torch.linalg.vector_norm(data.root_link_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(data.root_link_ang_vel_b, dim=-1)

    # Skip reset transients, tilted states and moving states.
    valid = (
        (elapsed > 3.0)
        & (tilt < torch.deg2rad(torch.tensor(5.0, device=env.device)))
        & (linear_speed < 0.03)
        & (angular_speed < 0.15)
        & torch.isfinite(height)
    )

    if not hasattr(env, "_standing_height_samples"):
        env._standing_height_samples = []

    env._standing_height_samples.extend(height[valid].detach().cpu().tolist())

    if len(env._standing_height_samples) >= 100:
        samples = torch.tensor(env._standing_height_samples)
        p10, median, p90 = torch.quantile(
            samples, torch.tensor([0.1, 0.5, 0.9])
        ).tolist()

        print(
            f"[standing height] median={median:.5f} m, p10={p10:.5f} m, p90={p90:.5f} m"
        )
        env._standing_height_samples.clear()


def make_microduck_standing_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:

    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    twist_cfg = cfg.commands.get("twist")

    assert isinstance(twist_cfg, UniformVelocityCommandCfg)

    twist_cfg.ranges.lin_vel_x = (
        -0.1,
        0.1,
    )

    twist_cfg.ranges.lin_vel_y = (
        -0.1,
        0.1,
    )

    twist_cfg.ranges.ang_vel_z = (
        -0.1,
        0.1,
    )

    twist_cfg.rel_standing_envs = 1.0
    twist_cfg.rel_heading_envs = 0.0
    twist_cfg.rel_world_envs = 0.0
    twist_cfg.rel_forward_envs = 0.0
    twist_cfg.init_velocity_prob = 0.0
    twist_cfg.rel_turn_in_place_envs = 0.0

    cfg.curriculum.clear()

    cfg.commands["head_pose"].ranges = (
        (0.0, 0.0),  # neck pitch
        (0.0, 0.0),  # head pitch
        (0.0, 0.0),  # head yaw
        (0.0, 0.0),  # head roll
    )

    cfg.commands["body_pose"].ranges = (
        (0.0, 0.0),  # x
        (0.0, 0.0),  # y
        (0.0, 0.0),  # z
        (0.0, 0.0),  # roll
        (0.0, 0.0),  # pitch
        (0.0, 0.0),  # yaw
    )

    for term in cfg.rewards.values():
        term.weight = 0.0

    # minimal rewards
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["pose"].weight = 1.0
    cfg.rewards["head_pose_tracking"].weight = 1.0

    # disbale disturbances
    cfg.events.pop("push_robot", None)
    cfg.events.pop("randomize_base_orientation", None)

    if play:
        cfg.events["report_standing_height"] = EventTermCfg(
            func=report_standing_height,
            mode="interval",
            interval_range_s=(0.1, 0.1),
        )

    return cfg
