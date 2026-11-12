"""Which reference foot targets does the stage-2 planner miss? Reads the ``FOOTHOLD_LOG=1`` output of
``scripts/acp/evaluate.py``.

A reference foot target counts when the tracker would get it (valid and inside the command's target footprint) on a
step the clip is still running; the planner hits it when it plans a target for that foot on the same step. Recall is
broken down by time to the reference contact onset, whether the target lies above the clip's ground (on an
obstacle), position in the scan, category, and the last second before a failure. The replay part repeats the
breakdown on the planner's own training labels (``<checkpoint>_footholds_replay.npz``), by target height relative to
the base, which both share.

Usage (numpy only)::

    python3 scripts/acp/analyze_footholds.py <run>/model_N_footholds_planner_phase0.0.npz [<run>/model_N_footholds_replay.npz]
"""

import sys
import numpy as np

ELEVATED_M = 0.08  # target this far above the clip's lowest foot target: on an obstacle
FAILURE_WINDOW_S = 1.0
FOOTPRINT = ((-0.8, 0.8), (-0.5, 0.5))
Z_BINS = [-np.inf, -0.8, -0.7, -0.6, -0.4, -0.2, np.inf]


def onset_time(contact: np.ndarray, live: np.ndarray, step_dt: float) -> np.ndarray:
    """``(T, N, F)`` seconds from each step to the next reference contact onset (rising edge) at or after it, NaN
    when none falls within the clip's logged steps."""
    steps, num_clips, num_feet = contact.shape
    previous = np.concatenate([np.zeros((1, num_clips, num_feet), bool), contact[:-1]])
    onset = contact & ~previous & live[..., None]
    out = np.full(contact.shape, np.nan)
    next_onset = np.full((num_clips, num_feet), np.nan)
    for t in range(steps - 1, -1, -1):
        next_onset = np.where(onset[t], t, next_onset)
        out[t] = (next_onset - t) * step_dt
    return out


def row(name: str, hit: np.ndarray, mask: np.ndarray, total: int) -> str:
    n = int(mask.sum())
    recall = hit[mask].mean() if n else float("nan")
    return f"  {name:28s} targets {n:7d} ({n / max(total, 1):6.1%})  recall {recall:6.1%}"


def binned(title, values, hit, mask, edges, fmt="{:.2f}"):
    print(title)
    total = int(mask.sum())
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = mask & (values >= lo) & (values < hi)
        print(row(f"[{fmt.format(lo)}, {fmt.format(hi)})", hit, sel, total))
    nan = mask & np.isnan(values)
    if nan.any():
        print(row("none in the logged steps", hit, nan, total))


def main(path: str, replay_path: str | None):
    d = np.load(path)
    live = d["live"].astype(bool)  # (T, N)
    ref_on = d["reference_on"].astype(bool) & live[..., None]  # (T, N, F)
    hit = d["planned_on"].astype(bool)
    target_b = d["reference_target_b"]
    target_w = d["reference_target_w"]
    planned_b = d["planned_b"]
    step_dt = float(d["step_dt"])
    failed, steps, clip_steps = d["failed"].astype(bool), d["steps"], d["clip_steps"]
    categories = d["categories"]
    num_steps, num_clips = live.shape
    total = int(ref_on.sum())
    print(f"{path}\n{num_clips} clips, {int(failed.sum())} failed; {total} reference foot targets on live steps")
    print(row("all", hit, ref_on, total))

    # above the clip's ground
    ground = np.where(d["reference_valid"].astype(bool) & live[..., None], target_w[..., 2], np.inf).min(axis=(0, 2))
    height = target_w[..., 2] - ground[None, :, None]
    elevated = height > ELEVATED_M
    print(f"target height above the clip's lowest foot target (> {ELEVATED_M} m = on an obstacle)")
    print(row("ground", hit, ref_on & ~elevated, total))
    print(row("obstacle", hit, ref_on & elevated, total))

    onset = onset_time(d["reference_contact"].astype(bool), live, step_dt)
    binned("time to the reference contact onset [s]", onset, hit, ref_on, [0, 0.1, 0.3, 0.6, 1.0, np.inf])
    binned("  on obstacles only", onset, hit, ref_on & elevated, [0, 0.1, 0.3, 0.6, 1.0, np.inf])

    x, y = target_b[..., 0], target_b[..., 1]
    binned("target x in the body-yaw frame [m]", x, hit, ref_on, [-0.8, -0.3, 0.0, 0.3, 0.5, 0.8001])
    binned("  on obstacles only", x, hit, ref_on & elevated, [-0.8, -0.3, 0.0, 0.3, 0.5, 0.8001])
    edge = np.minimum(FOOTPRINT[0][1] - np.abs(x), FOOTPRINT[1][1] - np.abs(y))
    binned("distance to the scan edge [m]", edge, hit, ref_on, [0, 0.1, 0.2, 0.3, np.inf])
    binned("target z in the body-yaw frame [m]", target_b[..., 2], hit, ref_on, Z_BINS)

    print("category")
    for c in sorted(set(categories.tolist())):
        sel = ref_on & (categories == c)[None, :, None]
        print(row(c, hit, sel, total) + f"   obstacle recall {hit[sel & elevated].mean() if (sel & elevated).any() else float('nan'):6.1%}")

    t = np.arange(num_steps)[:, None]
    before_failure = failed[None, :] & (t >= steps[None, :] - FAILURE_WINDOW_S / step_dt)
    window = before_failure[..., None]
    print(f"failed clips: last {FAILURE_WINDOW_S:.0f} s before the failure vs earlier; clips that succeed")
    print(row("failed, last second", hit, ref_on & window, total))
    print(row("failed, last second, obstacle", hit, ref_on & window & elevated, total))
    print(row("failed, earlier", hit, ref_on & failed[None, :, None] & ~window, total))
    print(row("succeeded", hit, ref_on & ~failed[None, :, None], total))
    print(row("succeeded, obstacle", hit, ref_on & ~failed[None, :, None] & elevated, total))
    fails_with_obstacle = (ref_on & window & elevated).any(axis=(0, 2))
    print(
        f"  failed clips with an obstacle foot target in their last second: {int(fails_with_obstacle.sum())}"
        f"/{int(failed.sum())}; of these, the planner missed it on every step of that second for"
        f" {int((fails_with_obstacle & ~(ref_on & window & elevated & hit).any(axis=(0, 2))).sum())}"
    )
    progress = steps / clip_steps
    print(f"  failed clips: median progress {np.median(progress[failed]) if failed.any() else float('nan'):.2f}")

    both = ref_on & hit
    error = planned_b - target_b
    for name, sel in (("ground", both & ~elevated), ("obstacle", both & elevated)):
        if sel.any():
            xy = np.linalg.norm(error[..., :2][sel], axis=-1)
            z = np.abs(error[..., 2][sel])
            print(
                f"position error where both have a target, {name}: xy median {np.median(xy):.3f} m"
                f" (90 % {np.quantile(xy, 0.9):.3f}), |z| median {np.median(z):.3f} m (90 % {np.quantile(z, 0.9):.3f})"
            )

    if replay_path is not None:
        r = np.load(replay_path)
        label_on = r["label_on"].astype(bool)
        r_hit = r["planned_on"].astype(bool)
        r_total = int(label_on.sum())
        print(f"\n{replay_path}\nplanner on {label_on.shape[0]} replay samples (its training labels), feet")
        print(row("all labels", r_hit, label_on, r_total))
        print(f"  precision {label_on[r_hit].mean():6.1%}")
        binned("label z in the body-yaw frame [m]", r["label_b"][..., 2], r_hit, label_on, Z_BINS)
        binned("label x in the body-yaw frame [m]", r["label_b"][..., 0], r_hit, label_on, [-0.8, -0.3, 0.0, 0.3, 0.5, 0.8001])
        r_both = label_on & r_hit
        r_err = r["planned_b"] - r["label_b"]
        xy = np.linalg.norm(r_err[..., :2][r_both], axis=-1)
        z = np.abs(r_err[..., 2][r_both])
        print(f"position error to the label: xy median {np.median(xy):.3f} m, |z| median {np.median(z):.3f} m")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
