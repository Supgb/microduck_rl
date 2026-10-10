import torch

from dataclasses import dataclass
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class TargetTrackingCommandCfg(CommandTermCfg):
   target_name: str = "target"
   entity_name: str = "robot"
   v_max: float = 0.35
   d_stop: float = 0.20
   sigma: float = 0.25
   k_yaw: float = 1.5
   max_ang_vel: float = 1.0
   resampling_time_range: tuple[float, float] = (1e6, 1e6) # disable resample

   def build(self, env: ManagerBasedRlEnv) -> "TargetTrackingCommand":
    return TargetTrackingCommand(self, env)


class TargetTrackingCommand(CommandTerm):
    cfg: TargetTrackingCommandCfg

    def __init__(self, cfg: TargetTrackingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.target: Entity = env.scene[cfg.target_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        # Must be registered here: `self.metrics` starts as an empty dict.
        self.metrics["dist_final"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        rel_w = (
            self.target.data.root_link_pos_w[:, :2]
            - self.robot.data.root_link_pos_w[:, :2]
        )
        dist = torch.linalg.norm(rel_w, dim=-1)
        # Overwritten every step -> the value read at reset is the episode's last distance.
        self.metrics["dist_final"][:] = dist
        # Latch: 1.0 once the robot has ever been within d_stop this episode.
        self.metrics["success"][:] = torch.maximum(
            self.metrics["success"], (dist <= self.cfg.d_stop).float()
        )

    def _resample_command(self,env_ids: torch.Tensor) -> None:
        pass

    def _update_command(self) -> None:
        """update [vx, vy, omega_z]"""

        # relative distance in world frame
        target_pos_w = self.target.data.root_link_pos_w[:, :2]
        robot_pos_w = self.robot.data.root_link_pos_w[:, :2]
        rel_w = target_pos_w - robot_pos_w
        dist = torch.linalg.norm(rel_w, dim=-1) 

        # project it into body frame
        heading = self.robot.data.heading_w
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        dx_b = cos_h * rel_w[:, 0] + sin_h * rel_w[:, 1]
        dy_b = -sin_h * rel_w[:, 0] + cos_h * rel_w[:, 1]

        # yaw
        heading_err = torch.atan2(dy_b, dx_b)

        dist_excess = torch.clamp(dist - self.cfg.d_stop, min=0.0)
        v_desired = self.cfg.v_max * torch.tanh(dist_excess / self.cfg.sigma)

        align_gate = torch.clamp(torch.cos(heading_err), min=0.0)
        self._command[:, 0] = v_desired * align_gate
        self._command[:, 1] = 0.0
        self._command[:, 2] = torch.clamp(
            self.cfg.k_yaw * heading_err, 
            min=-self.cfg.max_ang_vel, 
            max=self.cfg.max_ang_vel,
        )

        stopped = dist <= self.cfg.d_stop
        self._command[stopped, 2] = 0.0