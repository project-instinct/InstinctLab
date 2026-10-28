from __future__ import annotations

import gymnasium as gym
import torch
from typing import TYPE_CHECKING, Dict

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

from instinct_rl.env import VecEnv


class InstinctRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for Instinct-RL library
    Reference:
       https://github.com/project-instinct/instinct_rl/blob/master/instinct_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            num_obs_dims = self.unwrapped.observation_manager.group_obs_term_dim["policy"]
            num_obs_dims = [torch.prod(torch.tensor(dim, device="cpu")).item() for dim in num_obs_dims]
            self.num_obs = int(sum(num_obs_dims))
        else:
            # Not checked for DiectRlEnv
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            num_obs_dims = self.unwrapped.observation_manager.group_obs_term_dim["critic"]
            num_obs_dims = [torch.prod(torch.tensor(dim, device="cpu")).item() for dim in num_obs_dims]
            self.num_critic_obs = int(sum(num_obs_dims))
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            # Not checked for DiectRlEnv
            self.num_critic_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_critic_obs = None
        # reset at the start since the Instinct-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> ManagerBasedRLEnvCfg:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_pack = self.unwrapped.observation_manager.compute()
        else:
            obs_pack = self.unwrapped._get_observations()
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        return obs_pack["policy"], {"observations": obs_pack}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    @property
    def num_rewards(self) -> int:
        return self.unwrapped.num_rewards

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_pack, _ = self.env.reset()
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        # return observations
        return obs_pack["policy"], {"observations": obs_pack}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_pack, rew, terminated, truncated, extras = self.env.step(actions)
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict

        extras["observations"] = obs_pack
        # With ``compute_final_obs`` the env also reports the observations of finished episodes from
        # before their reset. Popped so a step without resets cannot leave a stale one behind.
        final_obs = extras.pop("final_obs", None)
        if final_obs is not None:
            extras["final_observations"] = self._flatten_all_obs_groups(final_obs)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        obs = obs_pack["policy"]
        if isinstance(rew, dict):
            # returned by multi-reward manager
            rew = self._stack_rewards(rew)
        else:
            # returned by regular reward manager
            # make sure rewards are always in shape of [batch, num_rewards]
            rew = rew.unsqueeze(1)
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Operations -- Instinct-RL
    """

    def get_obs_segments(self, group_name: str = "policy"):
        obs_term_names = self.unwrapped.observation_manager.active_terms[group_name]
        obs_term_dims = self.unwrapped.observation_manager.group_obs_term_dim[group_name]
        return self._get_obs_segments(obs_term_names, obs_term_dims)

    def _get_obs_segments(self, obs_term_names, obs_term_dims):
        # assuming the computed obs_term_dim is in the same order as the obs_cfg
        # From Python 3.6+, dictionaries are ordered by insertion order
        obs_segments = dict()
        for term_name, term_dim in zip(obs_term_names, obs_term_dims):
            obs_segments[term_name] = term_dim

        return obs_segments

    def get_obs_format(self) -> dict[str, dict[str, tuple]]:
        """Returns the observation information for all observation groups.
        Using this interface, so that, the algorithm / policy should not access env directly.
        But let the runner access env to get critical information.
        """
        obs_format = dict()
        for group_name in self.unwrapped.observation_manager.active_terms.keys():
            obs_format[group_name] = self.get_obs_segments(group_name)
        return obs_format

    """
    Internal Helpers
    """

    def _flatten_obs_group(self, obs_group: dict) -> torch.Tensor:
        """Considering observation_manager only concatenate observation terms of 1D tensors,
        this function flattens the observation terms of different shape and concatenate into
        a single tensor.
        """
        obs = []
        for obs_term, obs_value in obs_group.items():
            obs.append(obs_value.flatten(start_dim=1))
        obs = torch.cat(obs, dim=1)
        return obs

    def _flatten_all_obs_groups(self, obs_pack: dict) -> dict:
        obs_pack_ = dict()
        for obs_group_name, obs_group in obs_pack.items():
            obs_pack_[obs_group_name] = self._flatten_obs_group(obs_group) if isinstance(obs_group, dict) else obs_group
        return obs_pack_

    def _stack_rewards(self, rewards_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        rewards = []
        for reward_term, reward_value in rewards_dict.items():
            rewards.append(reward_value)
        rewards = torch.stack(rewards, dim=-1)
        return rewards
