#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _segment_map(z: np.lib.npyio.NpzFile) -> list[tuple[str, int, int]]:
    names = [str(x) for x in z["segment_names"].tolist()]
    starts = z["segment_starts"].astype(np.int64).tolist()
    ends = z["segment_ends"].astype(np.int64).tolist()
    return list(zip(names, starts, ends))


def _fmt_vec(x: np.ndarray, n: int = 8) -> str:
    flat = np.asarray(x).reshape(-1)
    head = flat[: min(n, flat.size)]
    return np.array2string(head, precision=5, separator=", ")


def _compare_array(name: str, a: np.ndarray, b: np.ndarray) -> None:
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        print(f"{name}: no rows")
        return
    aa = np.asarray(a[:n], dtype=np.float64)
    bb = np.asarray(b[:n], dtype=np.float64)
    diff = aa - bb
    print(
        f"{name}: rows={n}, shape={aa.shape[1:]}, "
        f"max_abs={np.max(np.abs(diff)):.6g}, mean_abs={np.mean(np.abs(diff)):.6g}, "
        f"rms={np.sqrt(np.mean(diff * diff)):.6g}"
    )
    print(f"  play[0] {_fmt_vec(aa[0])}")
    print(f"  sim [0] {_fmt_vec(bb[0])}")
    print(f"  diff[0] {_fmt_vec(diff[0])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare play.py and sim2sim obs debug NPZ dumps.")
    parser.add_argument("play_npz", type=Path)
    parser.add_argument("sim_npz", type=Path)
    parser.add_argument("--normalized", action="store_true", help="Compare obs_normalized instead of obs_raw.")
    args = parser.parse_args()

    play = np.load(args.play_npz, allow_pickle=True)
    sim = np.load(args.sim_npz, allow_pickle=True)
    obs_key = "obs_normalized" if args.normalized else "obs_raw"

    print(f"Comparing {obs_key}")
    _compare_array("full_obs", play[obs_key], sim[obs_key])
    print()

    play_segments = _segment_map(play)
    sim_segments = _segment_map(sim)
    if [(n, e - s) for n, s, e in play_segments] != [(n, e - s) for n, s, e in sim_segments]:
        print("Segment layouts differ:")
        print("  play", [(n, e - s) for n, s, e in play_segments])
        print("  sim ", [(n, e - s) for n, s, e in sim_segments])
        return

    for name, start, end in play_segments:
        _compare_array(name, play[obs_key][:, start:end], sim[obs_key][:, start:end])

    print()
    for key in (
        "command",
        "projected_gravity_b",
        "root_ang_vel_b",
        "root_quat_w",
        "root_pos_w",
        "joint_pos",
        "joint_vel",
        "actions",
        "applied_actions",
    ):
        if key in play and key in sim:
            _compare_array(key, play[key], sim[key])


if __name__ == "__main__":
    main()
