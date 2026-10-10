#!/usr/bin/env python3

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import mjlab_microduck.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"


def rising_edge_indices(x: np.ndarray) -> np.ndarray:
    """0 -> 1 transitions."""
    return np.flatnonzero((~x[:-1]) & x[1:]) + 1


def falling_edge_indices(x: np.ndarray) -> np.ndarray:
    """1 -> 0 transitions."""
    return np.flatnonzero(x[:-1] & (~x[1:])) + 1


def mean_or_nan(x):
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def main():
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--wandb-run-path",
        type=str,
    )
    source.add_argument(
        "--checkpoint-file",
        type=str,
    )

    parser.add_argument(
        "--wandb-checkpoint-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out",
        type=str,
        default="gait_phase52",
    )

    args = parser.parse_args()

    configure_torch_backends()

    device = args.device or (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------
    # 1. Load exactly the registered MicroDuck velocity task.
    # ------------------------------------------------------------

    env_cfg = load_env_cfg(
        TASK_ID,
        play=True,
    )
    agent_cfg = load_rl_cfg(TASK_ID)

    env_cfg.scene.num_envs = 1

    # ------------------------------------------------------------
    # 2. Make this a CLEAN gait experiment.
    # ------------------------------------------------------------

    # play=True normally gives a push every 0.5-1.0 s.
    # That is useful for robustness visualization but wrong for
    # nominal gait measurement.
    env_cfg.events.pop("push_robot", None)

    # Freeze the twist command.
    twist_cfg = env_cfg.commands["twist"]

    twist_cfg.resampling_time_range = (
        1000.0,
        1000.0,
    )

    twist_cfg.ranges.lin_vel_x = (
        args.vx,
        args.vx,
    )
    twist_cfg.ranges.lin_vel_y = (
        0.0,
        0.0,
    )
    twist_cfg.ranges.ang_vel_z = (
        0.0,
        0.0,
    )

    # Disable special command buckets.
    for attr in (
        "rel_standing_envs",
        "rel_heading_envs",
        "rel_turn_in_place_envs",
        "rel_world_envs",
        "rel_forward_envs",
        "init_velocity_prob",
    ):
        if hasattr(twist_cfg, attr):
            setattr(twist_cfg, attr, 0.0)

    # Freeze head/body command at zero.
    for name in ("head_pose", "body_pose"):
        pose_cfg = env_cfg.commands[name]

        pose_cfg.resampling_time_range = (
            1000.0,
            1000.0,
        )

        pose_cfg.ranges = tuple(
            (0.0, 0.0)
            for _ in pose_cfg.ranges
        )

    # Curricula are irrelevant during evaluation and could otherwise
    # change the command distributions.
    for name in (
        "standing_envs",
        "head_pose_range",
        "body_pose_range",
    ):
        env_cfg.curriculum.pop(name, None)

    # ------------------------------------------------------------
    # 3. Build environment + load trained actor.
    # ------------------------------------------------------------

    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
    )

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    if args.checkpoint_file is not None:
        resume_path = Path(
            args.checkpoint_file
        ).resolve()
    else:
        log_root = (
            Path("logs")
            / "rsl_rl"
            / agent_cfg.experiment_name
        ).resolve()

        resume_path, _ = get_wandb_checkpoint_path(
            log_root,
            Path(args.wandb_run_path),
            args.wandb_checkpoint_name,
        )

    print(
        f"[INFO] checkpoint = {resume_path}"
    )

    runner_cls = (
        load_runner_cls(TASK_ID)
        or MjlabOnPolicyRunner
    )

    runner = runner_cls(
        env,
        asdict(agent_cfg),
        device=device,
    )

    runner.load(
        str(resume_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )

    policy = runner.get_inference_policy(
        device=device
    )

    # ------------------------------------------------------------
    # 4. Get simulator truth used ONLY for analysis.
    # ------------------------------------------------------------

    base_env = env.unwrapped

    robot = base_env.scene["robot"]

    contact_sensor = (
        base_env.scene.sensors[
            "feet_ground_contact"
        ]
    )

    height_sensor = (
        base_env.scene[
            "foot_height_scan"
        ]
    )

    site_ids, site_names = robot.find_sites(
        ("left_foot", "right_foot"),
        preserve_order=True,
    )

    print(
        "[INFO] foot sites:",
        list(zip(site_names, site_ids)),
    )

    dt = float(base_env.step_dt)

    print(
        f"[INFO] control dt = {dt:.4f} s "
        f"({1.0 / dt:.1f} Hz)"
    )

    # ------------------------------------------------------------
    # 5. Rollout.
    # ------------------------------------------------------------

    obs = env.get_observations()

    n_steps = int(args.seconds / dt)

    rows = []

    previous_foot_pos = (
        robot.data.site_pos_w[
            0,
            site_ids,
            :,
        ]
        .detach()
        .clone()
    )

    for step in range(n_steps):

        with torch.no_grad():
            actions = policy(obs)

        obs, _, dones, _ = env.step(actions)

        # Break before an auto-reset contaminates the trace.
        if bool(dones[0].item()):
            print(
                f"[WARN] episode ended at "
                f"t={step * dt:.3f} s"
            )
            break

        # --------------------------
        # Contact state
        # --------------------------

        found = (
            contact_sensor.data.found[
                0, :2
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        left_contact = found[0] > 0
        right_contact = found[1] > 0

        # --------------------------
        # Contact / air timers
        # --------------------------

        current_contact_time = (
            contact_sensor.data.current_contact_time[
                0, :2
            ]
            .detach()
            .cpu()
            .numpy()
        )

        current_air_time = (
            contact_sensor.data.current_air_time[
                0, :2
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------
        # Foot terrain clearance
        # --------------------------

        foot_height = (
            height_sensor.data.heights[
                0, :2
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------
        # Foot world velocity
        #
        # We deliberately use finite differences instead of
        # assuming a site_vel_w API.
        # --------------------------

        foot_pos = (
            robot.data.site_pos_w[
                0,
                site_ids,
                :,
            ]
            .detach()
            .clone()
        )

        foot_vel_w = (
            foot_pos
            - previous_foot_pos
        ) / dt

        previous_foot_pos = foot_pos

        foot_xy_speed = torch.linalg.vector_norm(
            foot_vel_w[:, :2],
            dim=-1,
        ).cpu().numpy()

        # --------------------------
        # Base velocity
        #
        # Use BODY-frame vx because twist command is body-frame.
        # --------------------------

        base_vx = float(
            robot.data.root_link_lin_vel_b[
                0, 0
            ].item()
        )

        command = (
            base_env.command_manager
            .get_command("twist")
        )

        command_vx = float(
            command[0, 0].item()
        )

        rows.append(
            {
                "t": step * dt,

                "left_contact":
                    int(left_contact),

                "right_contact":
                    int(right_contact),

                "left_contact_time":
                    float(
                        current_contact_time[0]
                    ),

                "right_contact_time":
                    float(
                        current_contact_time[1]
                    ),

                "left_air_time":
                    float(
                        current_air_time[0]
                    ),

                "right_air_time":
                    float(
                        current_air_time[1]
                    ),

                "left_height":
                    float(foot_height[0]),

                "right_height":
                    float(foot_height[1]),

                "left_xy_speed":
                    float(foot_xy_speed[0]),

                "right_xy_speed":
                    float(foot_xy_speed[1]),

                "base_vx":
                    base_vx,

                "command_vx":
                    command_vx,
            }
        )

    env.close()

    if len(rows) < 2:
        raise RuntimeError(
            "Rollout too short."
        )

    # ------------------------------------------------------------
    # 6. Save raw CSV.
    # ------------------------------------------------------------

    csv_path = Path(
        f"{args.out}.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------
    # 7. Convert to arrays.
    # ------------------------------------------------------------

    t = np.array(
        [r["t"] for r in rows]
    )

    left = np.array(
        [
            bool(r["left_contact"])
            for r in rows
        ]
    )

    right = np.array(
        [
            bool(r["right_contact"])
            for r in rows
        ]
    )

    h_left = np.array(
        [r["left_height"] for r in rows]
    )

    h_right = np.array(
        [r["right_height"] for r in rows]
    )

    slip_left = np.array(
        [r["left_xy_speed"] for r in rows]
    )

    slip_right = np.array(
        [r["right_xy_speed"] for r in rows]
    )

    base_vx = np.array(
        [r["base_vx"] for r in rows]
    )

    cmd_vx = np.array(
        [r["command_vx"] for r in rows]
    )

    contact_time_left = np.array(
        [
            r["left_contact_time"]
            for r in rows
        ]
    )

    contact_time_right = np.array(
        [
            r["right_contact_time"]
            for r in rows
        ]
    )

    # ------------------------------------------------------------
    # 8. Ignore warm-up for gait statistics.
    # ------------------------------------------------------------

    keep = t >= args.warmup

    t_eval = t[keep]
    L = left[keep]
    R = right[keep]

    # ------------------------------------------------------------
    # 9. Gait events.
    # ------------------------------------------------------------

    left_td_idx = rising_edge_indices(L)    # left touchdown indices
    right_td_idx = rising_edge_indices(R)    # right touchdown indices

    left_lo_idx = falling_edge_indices(L)    # left lift-off indices
    right_lo_idx = falling_edge_indices(R)    # right lift-off indices

    left_td = t_eval[left_td_idx]    # left touchdown timestamps
    right_td = t_eval[right_td_idx]    # right touchdown timestamps

    left_lo = t_eval[left_lo_idx]    # left lift-off timestamps
    right_lo = t_eval[right_lo_idx]    # right lift-off timestamps

    # One stride = left touchdown -> next left touchdown.
    stride_periods = np.diff(left_td)

    mean_stride_period = mean_or_nan(
        stride_periods
    )

    stride_frequency = (
        1.0 / mean_stride_period
        if np.isfinite(mean_stride_period)
        else float("nan")
    )

    # Duty factor = fraction of time foot is in contact.
    duty_left = float(np.mean(L))
    duty_right = float(np.mean(R))

    # Contact-state fractions.
    double_support = float(
        np.mean(L & R)
    )

    left_single = float(
        np.mean(L & ~R)
    )

    right_single = float(
        np.mean(~L & R)
    )

    flight = float(
        np.mean(~L & ~R)
    )

    # ------------------------------------------------------------
    # 10. Left-right phase offset.
    #
    # For every left stride:
    #   phase = (right touchdown - left touchdown)
    #           / left stride period
    # ------------------------------------------------------------

    phases = []

    for start, end in zip(
        left_td[:-1],
        left_td[1:],
    ):
        candidates = right_td[
            (right_td > start)
            & (right_td < end)
        ]

        if len(candidates) > 0:
            phases.append(
                (candidates[0] - start)
                / (end - start)
            )

    phase_offset = mean_or_nan(phases)

    # ------------------------------------------------------------
    # 11. Stance-foot slip.
    #
    # Ignore the first 40 ms after touchdown, where impact /
    # settling can make finite-difference velocity large.
    # ------------------------------------------------------------

    settled_left = (
        keep
        & left
        & (contact_time_left > 0.04)
    )

    settled_right = (
        keep
        & right
        & (contact_time_right > 0.04)
    )

    left_stance_slip = mean_or_nan(
        slip_left[settled_left]
    )

    right_stance_slip = mean_or_nan(
        slip_right[settled_right]
    )

    # ------------------------------------------------------------
    # 12. Summary.
    # ------------------------------------------------------------

    print()
    print("========== GAIT SUMMARY ==========")

    print(
        f"mean base vx       : "
        f"{np.mean(base_vx[keep]):.3f} m/s"
    )

    print(
        f"command vx         : "
        f"{np.mean(cmd_vx[keep]):.3f} m/s"
    )

    print(
        f"stride period      : "
        f"{mean_stride_period:.3f} s"
    )

    print(
        f"stride frequency   : "
        f"{stride_frequency:.3f} Hz"
    )

    print(
        f"left duty factor   : "
        f"{duty_left:.3f}"
    )

    print(
        f"right duty factor  : "
        f"{duty_right:.3f}"
    )

    print(
        f"L->R phase offset  : "
        f"{phase_offset:.3f}"
    )

    print()
    print(
        f"double support     : "
        f"{double_support:.3f}"
    )

    print(
        f"left single        : "
        f"{left_single:.3f}"
    )

    print(
        f"right single       : "
        f"{right_single:.3f}"
    )

    print(
        f"flight             : "
        f"{flight:.3f}"
    )

    print()
    print(
        f"left stance slip   : "
        f"{left_stance_slip:.3f} m/s"
    )

    print(
        f"right stance slip  : "
        f"{right_stance_slip:.3f} m/s"
    )

    # ------------------------------------------------------------
    # 13. Plot.
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
    )

    axes[0].step(
        t,
        left.astype(float),
        where="post",
        label="left contact",
    )

    axes[0].step(
        t,
        right.astype(float),
        where="post",
        label="right contact",
    )

    axes[0].set_ylabel("contact")
    axes[0].legend()

    axes[1].plot(
        t,
        h_left,
        label="left foot",
    )

    axes[1].plot(
        t,
        h_right,
        label="right foot",
    )

    axes[1].axhline(
        0.02,
        linestyle="--",
        label="2 cm",
    )

    axes[1].set_ylabel("height [m]")
    axes[1].legend()

    axes[2].plot(
        t,
        base_vx,
        label="base vx",
    )

    axes[2].plot(
        t,
        cmd_vx,
        linestyle="--",
        label="command vx",
    )

    axes[2].set_ylabel("vx [m/s]")
    axes[2].legend()

    axes[3].plot(
        t,
        slip_left,
        label="left foot xy speed",
    )

    axes[3].plot(
        t,
        slip_right,
        label="right foot xy speed",
    )

    axes[3].set_ylabel("speed [m/s]")
    axes[3].set_xlabel("time [s]")
    axes[3].legend()

    fig.tight_layout()

    png_path = Path(
        f"{args.out}.png"
    )

    fig.savefig(
        png_path,
        dpi=160,
    )

    print()
    print(
        f"[INFO] saved {csv_path}"
    )
    print(
        f"[INFO] saved {png_path}"
    )


if __name__ == "__main__":
    main()