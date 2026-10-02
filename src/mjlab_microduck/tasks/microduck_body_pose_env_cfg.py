import copy
import math
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommandCfg,
)

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

MicroduckBodyPoseRlCfg = copy.deepcopy(MicroduckRlCfg)
MicroduckBodyPoseRlCfg.experiment_name = "body_pose"
MicroduckBodyPoseRlCfg.run_name = "pitch_tracking"
MicroduckBodyPoseRlCfg.logger = "tensorboard"
MicroduckBodyPoseRlCfg.upload_model = False


class PitchGuiCommand(microduck_mdp.UniformPoseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._manual_checkbox = None
        self._pitch_slider = None

    def create_gui(
        self,
        name,
        server,
        get_env_idx,
        on_change=None,
        request_action=None,
    ):
        del get_env_idx, on_change, request_action

        with server.gui.add_folder("Body pose"):
            self._manual_checkbox = server.gui.add_checkbox(
                "Manual pitch",
                initial_value=True,
            )
            self._pitch_slider = server.gui.add_slider(
                "Pitch (deg)",
                min=-15.0,
                max=15.0,
                step=0.5,
                initial_value=0.0,
            )

    def _apply_manual_command(self):
        if self._manual_checkbox is None:
            return

        if not self._manual_checkbox.value:
            return

        self._command.zero_()
        self._command[:, 4] = math.radians(self._pitch_slider.value)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)

        self._apply_manual_command()

    def _update_command(self):
        self._apply_manual_command()


@dataclass(kw_only=True)
class PitchGuiCommandCfg(microduck_mdp.UniformPoseCommandCfg):
    def build(self, env):
        return PitchGuiCommand(self, env)


def make_microduck_body_pose_env_cfg(
    nominal_height: float,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    cfg.curriculum.clear()
    twist_cfg = cfg.commands["twist"]
    assert isinstance(twist_cfg, UniformVelocityCommandCfg)

    twist_cfg.ranges.lin_vel_x = (-0.1, 0.1)
    twist_cfg.ranges.lin_vel_y = (-0.1, 0.1)
    twist_cfg.ranges.ang_vel_z = (-0.1, 0.1)

    twist_cfg.rel_standing_envs = 1.0
    twist_cfg.rel_heading_envs = 0.0
    twist_cfg.rel_world_envs = 0.0
    twist_cfg.rel_forward_envs = 0.0
    twist_cfg.rel_turn_in_place_envs = 0.0
    twist_cfg.init_velocity_prob = 0.0

    body_cfg = cfg.commands["body_pose"]
    body_cfg.ranges = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (-math.radians(15), math.radians(15)),
        (0.0, 0.0),
    )
    body_cfg.resampling_time_range = (4.0, 6.0)
    body_cfg.zero_command_prob = 0.3

    for term in cfg.rewards.values():
        term.weight = 0.0

    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_locomotion,
        weight=3.0,
        params={
            "command_name": "body_pose",
            "nominal_height": nominal_height,
            "xy_std": 0.02,
            "z_std": 0.02,
            "angle_std": math.radians(5),
            "axis_weights": (0.0, 0.0, 0.5, 1.0, 1.0, 0.0),
        },
    )

    cfg.rewards["pose"].weight = 0.0
    cfg.rewards["head_pose_tracking"].weight = 1.0

    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["self_collisions"].weight = -1.0

    cfg.events.pop("push_robot", None)
    cfg.events.pop("randomize_base_orientation", None)

    if play:
        original_cfg = cfg.commands["body_pose"]

        cfg.commands["body_pose"] = PitchGuiCommandCfg(
            resampling_time_range=original_cfg.resampling_time_range,
            ranges=original_cfg.ranges,
            zero_command_prob=original_cfg.zero_command_prob,
            debug_vis=original_cfg.debug_vis,
        )
    return cfg
