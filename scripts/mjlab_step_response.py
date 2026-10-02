import argparse
import csv
import math
from curses import raw
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import mdp as microduck_mdp

TASK_ID = "Mjlab-StandUp-Flat-MicroDuck"

PHYSICS_DT = 0.005
STEP_TIME = 8.5
DURATION = 10.0

ROOT_Z = 0.12
VIN = 7.4


class StepResponseLogger:
    def __init__(self, env, policy):
        self.raw_env = env
        self.policy = policy

        self.robot = env.scene["robot"]

        self.contact_sensor = env.scene["feet_ground_contact"]
        primary_cfg = self.contact_sensor.cfg.primary
        _, contact_names = self.robot.find_geoms(primary_cfg.pattern)
        contact_names = list(contact_names)
        print("Contact primaries:", contact_names)

        self.left_foot_idx = contact_names.index("left_foot_collision")
        self.right_foot_idx = contact_names.index("right_foot_collision")

        self.knee_joint_idx = self.robot.joint_names.index("left_knee")
        self.knee_dof_idx = int(
            self.robot.indexing.joint_v_adr[self.knee_joint_idx].item()
        )
        self.rows = []

    def log_after_step(self, action, command_time):
        contact_data = self.contact_sensor.data
        assert contact_data.found is not None
        assert contact_data.force is not None
        left_found = contact_data.found[0, self.left_foot_idx].item()
        left_force = contact_data.force[0, self.left_foot_idx]
        left_fz = float(left_force[2].item())
        left_fn = abs(left_fz)
        left_ft = float(torch.linalg.vector_norm(left_force[:2]).item())

        right_found = contact_data.found[0, self.right_foot_idx].item()
        right_force = contact_data.force[0, self.right_foot_idx]
        right_fz = float(right_force[2].item())
        right_fn = abs(right_fz)
        right_ft = float(torch.linalg.vector_norm(right_force[:2]).item())

        friction_ratio_left = left_ft / max(left_fn, 1e-8)
        friction_ratio_right = right_ft / max(right_fn, 1e-8)

        j = self.knee_joint_idx
        q = float(self.robot.data.joint_pos[0, j].item())
        qvel = float(self.robot.data.joint_vel[0, j].item())
        qacc = float(self.robot.data.joint_acc[0, j].item())
        tau = float(self.robot.data.qfrc_actuator[0, j].item())
        constraint_frc = float(
            self.raw_env.sim.data.qfrc_constraint[0, self.knee_dof_idx].item()
        )
        base_z = float(self.robot.data.root_link_pos_w[0, 2].item())
        raw_action = float(action[0, self.policy.knee_action_idx])
        target = (
            float(self.robot.data.default_joint_pos[0, j].item())
            + self.policy.knee_scale * raw_action
        )
        state_time = command_time + self.raw_env.step_dt
        command_error = target - q

        self.rows.append(
            (
                command_time,
                state_time,
                target,
                q,
                command_error,
                qvel,
                qacc,
                tau,
                constraint_frc,
                base_z,
                left_fn,
                left_ft,
                friction_ratio_left,
                left_found,
                right_fn,
                right_ft,
                friction_ratio_right,
                right_found,
            )
        )

    def save(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "time",
                    "state_time",
                    "target",
                    "q",
                    "command_error",
                    "qvel",
                    "qacc",
                    "tau",
                    "constraint_frc",
                    "base_z",
                    "left_fn",
                    "left_ft",
                    "friction_ratio_left",
                    "left_found",
                    "right_fn",
                    "right_ft",
                    "friction_ratio_right",
                    "right_found",
                ]
            )

            writer.writerows(self.rows)


class LoggingEnv:
    def __init__(self, env, logger):
        self._env = env
        self.logger = logger

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def get_observations(self):
        return self._env.get_observations()

    def reset(self):
        result = self._env.reset()
        self.logger.reset()
        return result

    def step(self, action):
        raw_env = self._env.unwrapped
        # Time at which this command is issued.
        command_time = raw_env.common_step_counter * raw_env.step_dt

        result = self._env.step(action)

        self.logger.log_after_step(action, command_time)

        return result

    def close(self):
        return self._env.close()


def make_diagnostic_cfg(delay_steps: int):
    # Deep-copied registered configuration.
    cfg = load_env_cfg(TASK_ID, play=True)

    cfg.seed = 0

    # -------------------------------------------------
    # Viser / mjlab 1.3.0 compatibility
    #
    # VelocityCommand.create_gui() assumes that the
    # positive command bound is >= 0.1 because its
    # "Max ..." slider uses:
    #
    #     min = 0.1
    #
    # StandUp normally uses:
    #
    #     vx   = ±0.01
    #     vy   = ±0.01
    #     yaw  = ±0.05
    #
    # which makes Viser fail during GUI construction.
    # -------------------------------------------------
    twist_cfg = cfg.commands.get("twist")

    if twist_cfg is not None:
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

        # Force the actual sampled velocity command
        # to zero for this diagnostic experiment.
        if hasattr(
            twist_cfg,
            "rel_standing_envs",
        ):
            twist_cfg.rel_standing_envs = 1.0

    # one robot
    cfg.scene.num_envs = 1

    # physics timing
    cfg.sim.mujoco.timestep = PHYSICS_DT
    cfg.decimation = 4
    cfg.episode_length_s = 20.0
    cfg.auto_reset = False

    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}

    for obs_group in cfg.observations.values():
        obs_group.enable_corruption = False

    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.init_state.pos = (0.0, 0.0, ROOT_Z)
    robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    joint_action_cfg = cfg.actions["joint_pos"]
    joint_action_cfg.scale = 1.0
    joint_action_cfg.use_default_offset = True

    bam_cfg = robot_cfg.articulation.actuators[0]
    bam_cfg.delay_min_lag = delay_steps
    bam_cfg.delay_max_lag = delay_steps

    if hasattr(bam_cfg, "vin_range"):
        bam_cfg.vin_range = (VIN, VIN)

    if hasattr(bam_cfg, "vin_drop_gain_range"):
        bam_cfg.vin_drop_gain_range = (0.0, 0.0)
    elif hasattr(bam_cfg, "vin_drop_resistance_range"):
        bam_cfg.vin_drop_resistance_range = (0.0, 0.0)

    cfg.events = {
        "expand_bam_friction_fields": EventTermCfg(
            func=(microduck_mdp.expand_bam_friction_fields), mode="startup"
        ),
        "reset_scene_to_default": EventTermCfg(
            func=reset_scene_to_default, mode="reset"
        ),
    }

    return cfg


class KneeStepPolicy:
    def __init__(self, env: ManagerBasedRlEnv, delta_q: float, step_time: float):
        self.env = env

        self.delta_q = delta_q
        self.step_time = step_time

        action_term = env.action_manager.get_term("joint_pos")
        if not isinstance(action_term, JointPositionAction):
            raise TypeError("joint_pos is not JointPositionAction")

        self.action_term = action_term
        print("Action targets:")

        for i, name in enumerate(action_term.target_names):
            print(f"{i:2d}: {name}")

        if "left_knee" not in action_term.target_names:
            raise RuntimeError("left_knee is not in action targets")

        self.knee_action_idx = action_term.target_names.index("left_knee")
        scale = action_term.scale

        if isinstance(scale, torch.Tensor):
            self.knee_scale = float(scale[0, self.knee_action_idx].item())
        else:
            self.knee_scale = float(scale)

        print(f"left_knee action index = {self.knee_action_idx}")
        print(f"left_knee action scale = {self.knee_scale}")

        self.knee_raw_step = self.delta_q / self.knee_scale

        print(f"requested delta-q = {self.delta_q:.4f} rad")
        print(f"raw action = {self.knee_raw_step:.4f}")

    def __call__(self, obs):
        del obs

        t = self.env.common_step_counter * self.env.step_dt

        action = torch.zeros(
            self.env.num_envs,
            self.env.action_manager.total_action_dim,
            device=self.env.device,
        )

        if t >= self.step_time:
            action[:, self.knee_action_idx] = self.knee_raw_step

        return action

    def reset(self):
        pass


def main(delta_q: float, delay_steps: int, duration: float):
    configure_torch_backends()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "mjlab/Mujoco Warp experiment should run on the CUDA machine"
        )

    device = "cuda:0"

    cfg = make_diagnostic_cfg(delay_steps)

    raw_env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
    logger = None

    try:
        print("\n===== Environment =====")
        print(f"physics dt = {raw_env.physics_dt}")
        print(f"environment dt = {raw_env.step_dt}")
        print(f"num envs = {raw_env.num_envs}")

        robot = raw_env.scene["robot"]
        print("\n===== Robot joints =====")
        for i, name in enumerate(robot.joint_names):
            print(f"{i:2d}: {name}")

        contact_sensor = raw_env.scene["feet_ground_contact"]
        primary_cfg = contact_sensor.cfg.primary
        _, contact_names = robot.find_geoms(primary_cfg.pattern)
        print("contact primaries:", list(contact_names))
        print("found shape:", contact_sensor.data.found.shape)
        print("force shape:", contact_sensor.data.force.shape)

        policy = KneeStepPolicy(raw_env, delta_q, STEP_TIME)
        logger = StepResponseLogger(raw_env, policy)
        logging_env = LoggingEnv(env, logger)
        viewer = ViserPlayViewer(logging_env, policy, frame_rate=60.0)

        num_steps = math.ceil(duration / raw_env.step_dt)

        print(f"\nRunning {num_steps} environment steps")
        print(f"Expected simulation duration: {duration:.2f} s")

        viewer.run(num_steps=num_steps)

    finally:
        if logger is not None:
            logger.save("mjlab_step_response.csv")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--delta-q", type=float, default=0.20)
    parser.add_argument("--delay-steps", type=int, default=0)
    parser.add_argument("--duration", type=float, default=DURATION)

    args = parser.parse_args()

    if args.delay_steps < 0:
        parser.error("--delay-steps must be >= 0")

    main(delta_q=args.delta_q, delay_steps=args.delay_steps, duration=args.duration)
