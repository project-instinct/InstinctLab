import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn

from instinctlab.managers import DummyRewardCfg, MultiRewardCfg, MultiRewardManager
from instinctlab.monitors import MonitorManager


class InstinctRlEnv(ManagerBasedRLEnv):
    """This class adds additional logging mechanism on sensors to get more
    comprehensive running statistics.
    """

    def load_managers(self):

        # check and routing the reward manager to the multi reward manager
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = DummyRewardCfg()
        super().load_managers()
        # replace the parent class's reward manager
        if "reward_group_cfg" in locals():
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = MultiRewardManager(self.cfg.rewards, self)
            print("[INFO] Multi-Reward Manager: ", self.reward_manager)

        self.monitor_manager = MonitorManager(self.cfg.monitors, self)
        print("[INFO] Monitor Manager: ", self.monitor_manager)

    def setup_manager_visualizers(self):
        managers = {
            "action_manager": self.action_manager,
            "observation_manager": self.observation_manager,
            "command_manager": self.command_manager,
            "termination_manager": self.termination_manager,
            "reward_manager": self.reward_manager,
            "curriculum_manager": self.curriculum_manager,
            "monitor_manager": self.monitor_manager,
        }

        def mean_reward(group_name: str | None = None) -> float:
            reward_buf = getattr(self, "reward_buf", None)
            if reward_buf is None:
                return 0.0
            return float(reward_buf[group_name].mean() if group_name is not None else reward_buf.mean())

        # the multi reward manager returns a dict of reward buffers, one per reward group
        if isinstance(self.reward_manager.active_terms, dict):
            reward_scalars = {
                f"mean_reward_{group_name}": lambda group_name=group_name: mean_reward(group_name)
                for group_name in self.reward_manager.active_terms
            }
        else:
            reward_scalars = {"mean_reward": mean_reward}
        scalars = {
            "episode": {
                **reward_scalars,
                "episode_length": lambda: float(self.episode_length_buf.float().mean()),
            }
        }
        for viz in self.sim.visualizers:
            viz.add_live_plots(managers, scalars=scalars)
        # Populate manager_visualizers for the Kit window (BaseEnvWindow reads this attribute).
        self.manager_visualizers = {
            name: mlv for v in self.sim.visualizers for name, mlv in getattr(v, "kit_manager_visualizers", {}).items()
        }

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        return_ = super().step(action)
        monitor_infos = self.monitor_manager.update(dt=self.step_dt)
        self.extras["step"] = self.extras.get("step", {})
        self.extras["step"].update(monitor_infos)
        return return_

    def _reset_idx(self, env_ids: torch.Tensor | slice):
        # the monitor manager indexes its buffers with an explicit tensor, so resolve the selector once
        monitor_env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        monitor_infos = self.monitor_manager.reset(monitor_env_ids, is_episode=True)
        return_ = super()._reset_idx(env_ids)
        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(monitor_infos)
        return return_

    """
    Properties.
    """

    @property
    def num_rewards(self) -> int:
        return getattr(self.reward_manager, "num_rewards", 1)
