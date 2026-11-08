"""An Instinct-RL environment that labels its own rollouts for a contact planner (ACP stage 2).

Runs a :class:`~instinctlab.tasks.shadowing.acp.hindsight_labeler.HindsightContactLabeler` configured by
``cfg.contact_labeler`` and puts the samples of the episodes that finished in a step into
``extras[cfg.contact_labeler.infos_key]`` (possibly empty), which reaches the algorithm as ``infos``.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.common import VecEnvStepReturn

from instinctlab.envs.manager_based_rl_env import InstinctRlEnv

from .contact_schedule import heading
from .hindsight_labeler import HindsightContactLabeler, HindsightContactLabelerCfg


class HindsightLabelingEnv(InstinctRlEnv):
    """:class:`InstinctRlEnv` with hindsight contact labeling (``cfg.contact_labeler``).

    The admission rule reads the reference phase of each episode: (motion start time + elapsed time) / clip length of
    the scene's single motion buffer.
    """

    def load_managers(self):
        super().load_managers()
        cfg: HindsightContactLabelerCfg = self.cfg.contact_labeler
        self._labeler_robot = self.scene[cfg.asset_name]
        self._labeler_sensor = self.scene.sensors[cfg.sensor_name]
        self._labeler_robot_body_ids = self._labeler_robot.find_bodies(cfg.body_names, preserve_order=True)[0]
        self._labeler_sensor_body_ids = self._labeler_sensor.find_bodies(cfg.body_names, preserve_order=True)[0]
        if not self._labeler_sensor.cfg.filter_prim_paths_expr:
            raise ValueError(
                f"the hindsight labeler needs {cfg.sensor_name!r} filtered to the terrain (filter_prim_paths_expr), so"
                " that its contacts exclude self-contacts"
            )
        motion_reference = self.scene[cfg.motion_reference_name]
        if len(motion_reference.motion_buffers) != 1:
            raise ValueError("the hindsight labeler's reference phase needs exactly one motion buffer")
        self._labeler_motion_buffer = next(iter(motion_reference.motion_buffers.values()))
        term_dims = self.observation_manager.group_obs_term_dim[cfg.obs_group]
        obs_size = int(sum(np.prod(dims) for dims in term_dims))
        self.contact_labeler = HindsightContactLabeler(cfg, self.num_envs, obs_size, self.step_dt, self.device)
        print(f"[INFO] Hindsight contact labeler: {cfg.num_slots} slots x {cfg.max_recorded_steps} steps")

    def reference_phase(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Fraction of the reference clip reached by each env's episode so far."""
        buffer = self._labeler_motion_buffer
        assigned = buffer.env_ids_to_assigned_ids(env_ids).to(buffer.buffer_device)
        motion_ids = buffer._assigned_env_motion_selection[assigned]
        sequences = buffer._all_motion_sequences
        clip_length_s = sequences.buffer_length[motion_ids] / sequences.framerate[motion_ids]
        elapsed_s = self.episode_length_buf[env_ids].to(assigned.device) * self.step_dt
        return ((buffer._motion_buffer_start_time_s[assigned] + elapsed_s) / clip_length_s).to(self.device)

    def _record_labeler_step(self):
        cfg: HindsightContactLabelerCfg = self.cfg.contact_labeler
        group = self.obs_buf[cfg.obs_group]
        planner_obs = (
            torch.cat([v.flatten(start_dim=1) for v in group.values()], dim=1) if isinstance(group, dict) else group
        )
        # vertical force from the terrain only: the net force also holds self-contacts (e.g. pelvis against thighs)
        force_z = self._labeler_sensor.data.force_matrix_w.torch[:, self._labeler_sensor_body_ids, :, 2].sum(dim=-1)
        robot = self._labeler_robot
        self.contact_labeler.record(
            planner_obs,
            force_z,
            robot.data.body_link_pos_w.torch[:, self._labeler_robot_body_ids],
            robot.data.root_pos_w.torch,
            heading(math_utils.yaw_quat(robot.data.root_quat_w.torch)),
        )

    def _reset_idx(self, env_ids: torch.Tensor | slice):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        # the finished episodes' data (episode length, motion start time) is intact until the reset below
        self.contact_labeler.finish_episodes(ids, self.episode_length_buf[ids], self.reference_phase(ids))
        return_ = super()._reset_idx(env_ids)
        self.contact_labeler.start_episodes(ids, self.reference_phase(ids))
        return return_

    def reset(self, *args, **kwargs):
        return_ = super().reset(*args, **kwargs)
        self._record_labeler_step()
        return return_

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        return_ = super().step(action)
        self._record_labeler_step()
        samples = self.contact_labeler.pop_samples()
        self.extras[self.cfg.contact_labeler.infos_key] = samples
        stats = self.contact_labeler.stats
        self.extras["step"].update(
            {
                "hindsight_samples": float(samples["planner_obs"].shape[0]),
                "hindsight_recording_slots": float(self.contact_labeler.num_recording),
                **{f"hindsight_{k}": float(v) for k, v in stats.items()},
            }
        )
        for k in stats:
            stats[k] = 0
        return return_
