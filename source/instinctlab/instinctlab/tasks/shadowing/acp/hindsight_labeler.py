"""Hindsight contact labeling of the policy's own rollouts (ACP stage 2, paper "Hindsight Contact Labeling").

The labeler records the trajectories of selected episodes (planner observation, per-body contact state, body
positions, base pose; at most ``max_recorded_steps`` steps) in a fixed pool of recording slots. When an episode ends
and passes the admission rule, it is scanned backward: for every recorded step and contact body, the next contact
onset at or after that step (contact segments shorter than ``min_contact_steps`` removed first) within the horizon
gives the target, expressed in that step's body-yaw frame and kept if it lies inside the scan footprint. Each
recorded step becomes one sample ``(planner_obs, target_pos, target_valid)`` for the planner's replay.

Admission (reference-following episodes): the episode started at a reference phase of at most
``start_phase_max`` and ended at a phase of at least ``end_phase_min``. Only episodes that can pass are recorded:
a recording slot is taken at the episode start when its start phase qualifies, so memory is bounded by
``num_slots`` and not by the number of environments. Eligible episodes without a free slot are not recorded.

Everything here is plain torch on tensors the environment hands in (see ``labeling_env.HindsightLabelingEnv``), so
it is tested without a simulator.
"""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass

#: Keys of one sample (cross-repo contract with Instinct-RL's ``PlannerAlgoMixin``).
SAMPLE_KEYS = ("planner_obs", "target_pos", "target_valid")
#: With ``partition_samples`` every sample also carries its replay partition and group (Instinct-RL's
#: ``PARTITIONED_SAMPLE_KEYS``).
PARTITION_KEYS = ("partition", "group")
#: With ``dual_buffer`` (ACP stage 3), every sample also carries ``buffer_id`` (0 = ref, 1 = goal) and ``quality``.
DUAL_BUFFER_KEYS = ("buffer_id", "quality")


def label_next_contacts(
    contact: torch.Tensor,
    body_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_yaw: torch.Tensor,
    length: torch.Tensor,
    horizon_steps: int,
    min_contact_steps: int,
    footprint: tuple[tuple[float, float], tuple[float, float]],
    min_gap_steps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-contact targets of recorded trajectories.

    Args:
        contact: (S, L, B) contact state of each body at each recorded step.
        body_pos_w: (S, L, B, 3) body positions in the world frame.
        root_pos_w: (S, L, 3) base positions in the world frame.
        root_yaw: (S, L) base yaw.
        length: (S,) number of valid steps of each trajectory; later steps are ignored.
        horizon_steps: An onset at most this many steps ahead is a target.
        min_contact_steps: Contact segments shorter than this are removed before the onset search.
        footprint: ``((x_min, x_max), (y_min, y_max))`` of kept targets in the body-yaw frame.
        min_gap_steps: Interior non-contact gaps shorter than this are filled before segment filtering.

    Returns:
        Targets (S, L, B, 3) in each step's body-yaw frame, zero where not valid, and their validity (S, L, B).
    """
    num_slots, num_steps, num_bodies = contact.shape
    device = contact.device
    steps = torch.arange(num_steps, device=device)
    in_range = steps.unsqueeze(0) < length.unsqueeze(1)  # (S, L)
    flags = (contact & in_range.unsqueeze(-1)).transpose(1, 2).reshape(-1, num_steps)  # (S * B, L)
    row_offset = torch.arange(flags.shape[0], device=device).unsqueeze(1) * (num_steps + 1)
    if min_gap_steps > 0:
        # fill interior non-contact gaps shorter than min_gap_steps before checking segment lengths
        gap = ~flags
        gap_previous = torch.cat([torch.zeros_like(gap[:, :1]), gap[:, :-1]], dim=1)
        gap_id = (torch.cumsum(gap & ~gap_previous, dim=1) + row_offset).flatten()
        gap_length = torch.zeros(flags.shape[0] * (num_steps + 1), dtype=torch.long, device=device)
        gap_length.index_add_(0, gap_id, gap.flatten().long())
        short_gap = gap & (gap_length[gap_id].reshape(gap.shape) < min_gap_steps)
        seen_contact = torch.cumsum(flags.long(), dim=1) > 0
        future_contact = torch.flip(torch.cumsum(torch.flip(flags.long(), dims=[1]), dim=1), dims=[1]) > 0
        flags = flags | (short_gap & seen_contact & future_contact)
    previous = torch.cat([torch.zeros_like(flags[:, :1]), flags[:, :-1]], dim=1)
    # drop short contact segments: number every segment of a row, count its steps, keep the long ones
    segment_id = torch.cumsum(flags & ~previous, dim=1)  # 1-based within the row, valid where flags
    global_id = (segment_id + row_offset).flatten()
    segment_length = torch.zeros(flags.shape[0] * (num_steps + 1), dtype=torch.long, device=device)
    segment_length.index_add_(0, global_id, flags.flatten().long())
    kept = flags & (segment_length[global_id].reshape(flags.shape) >= min_contact_steps)
    # next onset at or after each step: reverse cumulative minimum over the onset steps
    kept_previous = torch.cat([torch.zeros_like(kept[:, :1]), kept[:, :-1]], dim=1)
    onset_step = torch.where(
        kept & ~kept_previous, steps.unsqueeze(0), torch.full_like(kept, num_steps, dtype=torch.long)
    )
    next_onset = torch.flip(torch.cummin(torch.flip(onset_step, dims=[1]), dim=1).values, dims=[1])
    next_onset = next_onset.reshape(num_slots, num_bodies, num_steps).transpose(1, 2)  # (S, L, B)
    valid = (next_onset < num_steps) & (next_onset - steps.view(1, -1, 1) <= horizon_steps) & in_range.unsqueeze(-1)
    # onset positions in the body-yaw frame of each step
    onset_index = next_onset.clamp(max=num_steps - 1)
    slot_index = torch.arange(num_slots, device=device).view(-1, 1, 1)
    body_index = torch.arange(num_bodies, device=device).view(1, 1, -1)
    offset = body_pos_w[slot_index, onset_index, body_index] - root_pos_w.unsqueeze(2)  # (S, L, B, 3)
    cos, sin = torch.cos(root_yaw).unsqueeze(-1), torch.sin(root_yaw).unsqueeze(-1)
    local = torch.stack(
        [cos * offset[..., 0] + sin * offset[..., 1], -sin * offset[..., 0] + cos * offset[..., 1], offset[..., 2]],
        dim=-1,
    )
    (x_min, x_max), (y_min, y_max) = footprint
    inside = (local[..., 0] >= x_min) & (local[..., 0] <= x_max) & (local[..., 1] >= y_min) & (local[..., 1] <= y_max)
    valid = valid & inside
    return local * valid.unsqueeze(-1), valid


@configclass
class HindsightContactLabelerCfg:
    """Settings of :class:`HindsightContactLabeler` (defaults: ACP Supp. Table S5)."""

    obs_group: str = "planner"
    """Observation group recorded as the planner input."""

    body_names: list[str] = MISSING
    """Contact bodies, in the planner's output order."""

    asset_name: str = "robot"
    sensor_name: str = "contact_forces"
    motion_reference_name: str = "motion_reference"

    contact_source: Literal["robot", "reference"] = "robot"
    """Where the labeled contacts come from: ``"robot"`` (paper) the robot's terrain contacts and body positions;
    ``"reference"`` (experiment) the reference motion's contact flags and body positions, so the labels follow the
    reference contact schedule the tracker learned in stage 1."""

    force_threshold: float = 1.0
    """``"robot"`` contacts: a body is in contact when its terrain contact force norm exceeds this (N)."""

    min_gap_steps: int = 5
    """Interior non-contact gaps shorter than this many control steps are filled before segment filtering."""

    min_contact_steps: int = 10
    """Contact segments shorter than this many control steps are discarded."""

    horizon_s: float = 1.5
    """Contact lookahead horizon."""

    footprint: tuple[tuple[float, float], tuple[float, float]] = ((-0.8, 0.8), (-0.5, 0.5))
    """Kept targets lie in this ``((x_min, x_max), (y_min, y_max))`` box of the body-yaw frame (the planner scan)."""

    max_recorded_steps: int = 1000
    """Steps recorded per episode; later steps are not recorded (separate from the episode timeout)."""

    num_slots: int = 512
    """Episodes recorded at once (memory: ``num_slots * max_recorded_steps`` steps)."""

    start_phase_max: float = 0.1
    """Admission: the episode started at most at this fraction of the reference clip."""

    end_phase_min: float = 0.9
    """Admission: the episode reached at least this fraction of the reference clip."""

    label_unadmitted: bool = False
    """Experiment: also label the recorded episodes that end before ``end_phase_min`` (failures, mostly), up to the
    last step whose lookahead (horizon plus minimum contact length) lies inside the episode, so the steps just before
    the episode's end, whose next contacts are cut off by it, are dropped. The paper labels admitted episodes only."""

    partition_samples: bool = False
    """Experiment: the samples carry ``partition`` and ``group`` ids (the env passes them per episode, e.g. the
    reference clip and its terrain category) for a partitioned, group-balanced planner replay."""

    dual_buffer: bool = False
    """ACP stage 3: also record per-step positive mechanical work and emit ``buffer_id`` and ``quality`` keys for
    dual-buffer (reference + goal) replay routing and quality-based eviction."""

    contact_point: bool = False
    """Experiment (``"robot"`` contacts): label the terrain contact point the contact sensor reports instead of the body
    origin, so the labels mark the same point as the dataset's contact targets (the annotated terrain point)."""

    infos_key: str = "planner_samples"
    """Key of the samples in the step ``extras``."""


class HindsightContactLabeler:
    """Records eligible episodes in recording slots and turns admitted ones into planner samples.

    Call order per environment step: :meth:`finish_episodes` for the envs about to be reset (their episode data is
    still intact), :meth:`start_episodes` once they are reset, and :meth:`record` after the observations of the
    step are computed; :meth:`pop_samples` returns what the finished episodes produced.
    """

    def __init__(self, cfg: HindsightContactLabelerCfg, num_envs: int, obs_size: int, step_dt: float, device):
        self.cfg = cfg
        self.num_bodies = len(cfg.body_names)
        self.horizon_steps = int(round(cfg.horizon_s / step_dt))
        self.device = torch.device(device)
        num_slots, num_steps = cfg.num_slots, cfg.max_recorded_steps
        self.obs = torch.zeros(num_slots, num_steps, obs_size, device=self.device)
        self.contact = torch.zeros(num_slots, num_steps, self.num_bodies, dtype=torch.bool, device=self.device)
        self.body_pos_w = torch.zeros(num_slots, num_steps, self.num_bodies, 3, device=self.device)
        self.root_pos_w = torch.zeros(num_slots, num_steps, 3, device=self.device)
        self.root_yaw = torch.zeros(num_slots, num_steps, device=self.device)
        self.work = torch.zeros(num_slots, num_steps, device=self.device) if cfg.dual_buffer else None
        self.slot_length = torch.zeros(num_slots, dtype=torch.long, device=self.device)
        self.slot_env = torch.full((num_slots,), -1, dtype=torch.long, device=self.device)
        self.env_slot = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self._samples: list[dict[str, torch.Tensor]] = []
        self.stats = {
            "admitted_episodes": 0,
            "rejected_episodes": 0,
            "unrecorded_episodes": 0,
            "unadmitted_samples": 0,
        }
        self._empty = {
            "planner_obs": torch.zeros(0, obs_size, device=self.device),
            "target_pos": torch.zeros(0, self.num_bodies, 3, device=self.device),
            "target_valid": torch.zeros(0, self.num_bodies, dtype=torch.bool, device=self.device),
        }
        self.sample_keys = (
            SAMPLE_KEYS
            + (PARTITION_KEYS if cfg.partition_samples else ())
            + (DUAL_BUFFER_KEYS if cfg.dual_buffer else ())
        )
        if cfg.partition_samples:
            self._empty.update({k: torch.zeros(0, dtype=torch.long, device=self.device) for k in PARTITION_KEYS})
        if cfg.dual_buffer:
            self._empty["buffer_id"] = torch.zeros(0, dtype=torch.long, device=self.device)
            self._empty["quality"] = torch.zeros(0, dtype=torch.float, device=self.device)

    def start_episodes(
        self,
        env_ids: torch.Tensor,
        start_phase: torch.Tensor | None = None,
        eligible: torch.Tensor | None = None,
    ) -> None:
        """Give a free slot to every eligible env in ``env_ids``."""
        if eligible is None:
            if start_phase is None:
                raise ValueError("start_episodes requires either start_phase or eligible")
            eligible_mask = start_phase <= self.cfg.start_phase_max
        else:
            eligible_mask = eligible.bool()
        eligible_envs = env_ids[eligible_mask]
        free = (self.slot_env < 0).nonzero().squeeze(-1)
        num = min(len(eligible_envs), len(free))
        self.stats["unrecorded_episodes"] += len(eligible_envs) - num
        slots, envs = free[:num], eligible_envs[:num]
        self.slot_env[slots] = envs
        self.env_slot[envs] = slots
        self.slot_length[slots] = 0
        if self.work is not None and num > 0:
            self.work[slots] = 0.0

    def record(
        self,
        planner_obs: torch.Tensor,
        contact: torch.Tensor,
        body_pos_w: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_yaw: torch.Tensor,
        positive_work: torch.Tensor | None = None,
    ) -> None:
        """Append the current step of every recorded env (all inputs are per env: (N, ...); ``contact`` (N, B) bool)."""
        slots = ((self.slot_env >= 0) & (self.slot_length < self.cfg.max_recorded_steps)).nonzero().squeeze(-1)
        envs, steps = self.slot_env[slots], self.slot_length[slots]
        self.obs[slots, steps] = planner_obs[envs]
        self.contact[slots, steps] = contact[envs]
        self.body_pos_w[slots, steps] = body_pos_w[envs]
        self.root_pos_w[slots, steps] = root_pos_w[envs]
        self.root_yaw[slots, steps] = root_yaw[envs]
        if self.work is not None and positive_work is not None:
            self.work[slots, steps] = positive_work[envs]
        self.slot_length[slots] += 1

    def finish_episodes(
        self,
        env_ids: torch.Tensor,
        episode_length: torch.Tensor,
        end_phase: torch.Tensor | None = None,
        partition: torch.Tensor | None = None,
        group: torch.Tensor | None = None,
        admitted: torch.Tensor | None = None,
        cut_length: torch.Tensor | None = None,
        quality_denom: torch.Tensor | None = None,
        buffer_id: torch.Tensor | None = None,
    ) -> None:
        """Label the admitted episodes of ``env_ids`` and free their slots."""
        if self.cfg.partition_samples and (partition is None or group is None):
            raise ValueError("partition_samples needs the partition and group of every finished episode")
        if self.cfg.dual_buffer and buffer_id is None:
            raise ValueError("dual_buffer needs buffer_id for every finished episode")
        slots = self.env_slot[env_ids]
        recorded = slots >= 0
        slots = slots[recorded]
        episode_length = episode_length[recorded]
        if admitted is None:
            if end_phase is None:
                raise ValueError("finish_episodes requires either end_phase or admitted")
            admitted_mask = end_phase[recorded] >= self.cfg.end_phase_min
        else:
            admitted_mask = admitted.to(self.device)[recorded].bool()
        ids = None
        if self.cfg.partition_samples:
            ids = torch.stack([partition.to(self.device)[recorded], group.to(self.device)[recorded]], dim=-1).long()
        cut_rec = None if cut_length is None else cut_length.to(self.device)[recorded].long()
        denom_rec = None if quality_denom is None else quality_denom.to(self.device)[recorded].float()
        buf_rec = None if buffer_id is None else buffer_id.to(self.device)[recorded].long()

        self.stats["admitted_episodes"] += int(admitted_mask.sum())
        self.stats["rejected_episodes"] += int((~admitted_mask).sum())
        if bool(admitted_mask.any()):
            self._samples.append(
                self._label(
                    slots[admitted_mask],
                    episode_length[admitted_mask],
                    ids=None if ids is None else ids[admitted_mask],
                    cut_length=None if cut_rec is None else cut_rec[admitted_mask],
                    quality_denom=None if denom_rec is None else denom_rec[admitted_mask],
                    buffer_id=None if buf_rec is None else buf_rec[admitted_mask],
                )
            )
        if self.cfg.label_unadmitted and bool((~admitted_mask).any()):
            unadmitted = ~admitted_mask
            samples = self._label(
                slots[unadmitted],
                episode_length[unadmitted],
                cut_tail=True,
                ids=None if ids is None else ids[unadmitted],
                quality_denom=None if denom_rec is None else denom_rec[unadmitted],
                buffer_id=None if buf_rec is None else buf_rec[unadmitted],
            )
            self.stats["unadmitted_samples"] += samples["planner_obs"].shape[0]
            self._samples.append(samples)
        self.slot_env[slots] = -1
        self.env_slot[env_ids[recorded]] = -1

    def _label(
        self,
        slots: torch.Tensor,
        episode_length: torch.Tensor,
        cut_tail: bool = False,
        ids: torch.Tensor | None = None,
        cut_length: torch.Tensor | None = None,
        quality_denom: torch.Tensor | None = None,
        buffer_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Samples of the recorded ``slots``."""
        length = self.slot_length[slots]
        target_pos, target_valid = label_next_contacts(
            self.contact[slots],
            self.body_pos_w[slots],
            self.root_pos_w[slots],
            self.root_yaw[slots],
            length,
            self.horizon_steps,
            self.cfg.min_contact_steps,
            self.cfg.footprint,
            min_gap_steps=self.cfg.min_gap_steps,
        )
        # an episode longer than its record hides the onsets after the record's end: drop the steps whose
        # horizon (plus the segment-length check) reaches past it
        truncated = (episode_length > length) | cut_tail
        usable = torch.where(truncated, length - self.horizon_steps - self.cfg.min_contact_steps + 1, length)
        if cut_length is not None:
            has_cut = cut_length >= 0
            # cut_length is the 0-based step index where d_g < 0.2 m; emit steps [0, cut_length] inclusive
            usable = torch.where(has_cut, torch.minimum(usable, cut_length + 1), usable)
        usable = usable.clamp_min(0)
        steps = torch.arange(self.cfg.max_recorded_steps, device=self.device).unsqueeze(0)
        rows = steps < usable.unsqueeze(1)
        samples = {
            "planner_obs": self.obs[slots][rows],
            "target_pos": target_pos[rows],
            "target_valid": target_valid[rows],
        }
        if ids is not None:
            per_step = ids.unsqueeze(1).expand(-1, rows.shape[1], -1)[rows]
            samples["partition"], samples["group"] = per_step[:, 0], per_step[:, 1]
        if self.cfg.dual_buffer:
            buf_per_step = buffer_id.unsqueeze(1).expand(-1, rows.shape[1])[rows]
            samples["buffer_id"] = buf_per_step
            # D12: quality = -sum_{t < usable} work[t] / max(d_init, 0.5)
            ep_work = (self.work[slots] * rows.float()).sum(dim=1)
            denom = torch.ones_like(ep_work) if quality_denom is None else quality_denom.clamp_min(0.5)
            ep_quality = -ep_work / denom
            samples["quality"] = ep_quality.unsqueeze(1).expand(-1, rows.shape[1])[rows]
        return samples

    def pop_samples(self) -> dict[str, torch.Tensor]:
        """Samples of the episodes finished since the last call (K may be 0)."""
        samples = self._samples or [self._empty]
        self._samples = []
        return {k: torch.cat([s[k] for s in samples]) for k in self.sample_keys}

    @property
    def num_recording(self) -> int:
        return int((self.slot_env >= 0).sum())
