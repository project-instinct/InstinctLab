"""How much GPU memory does the simulator take per environment?

``num_envs`` and the learner's batch sizes are separate memory budgets: everything the learner
allocates is shaped ``(horizon, num_imagined, ...)`` and does not grow with the number of parallel
environments, while the simulator's constraint and contact buffers are allocated per environment.
This measures the second one in isolation, by building the env at a series of sizes and reading the
allocator's high-water mark after a reset.

Reports both the torch allocator's peak and the driver's total, because the physics backend allocates
through Warp rather than through the caching allocator, so torch's counter alone understates it.
"""

import gc
import os
import sys

import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
SIZES = [int(x) for x in os.environ.get("PROBE_SIZES", "128,512,1024,2048,4096").split(",")]


def driver_used_gib() -> float:
    """Memory the driver reports as in use, which includes allocation outside torch."""
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().split("\n")[0]
    return float(out) / 1024.0


parser = cli.build_train_parser()
results = []
for n in SIZES:
    args_cli, overrides = cli.parse_cli(
        parser, argv=["--task", TASK, "--num_envs", str(n), "--run_name", f"probe{n}"], require_run_name=True
    )
    from isaaclab.app import launch_simulation

    env_cfg, _ = cli.resolve_train_cfg(TASK, overrides)
    env_cfg.scene.num_envs = n
    before = driver_used_gib()
    try:
        with launch_simulation(env_cfg, args_cli):
            import gymnasium as gym

            sim = gym.make(TASK, cfg=env_cfg)
            sim.reset()
            torch.cuda.synchronize()
            peak_torch = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            in_use = driver_used_gib()
            print(
                f"RESULT num_envs={n:5d} torch_peak={peak_torch:6.2f}GiB "
                f"torch_reserved={peak_reserved:6.2f}GiB driver={in_use:6.2f}GiB "
                f"delta_over_baseline={in_use - before:6.2f}GiB",
                flush=True,
            )
            results.append((n, peak_torch, peak_reserved, in_use, before))
            sim.close()
            del sim
    except Exception as exc:  # OOM or backend limit
        print(f"RESULT num_envs={n:5d} FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        results.append((n, None, None, None, before))
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

print("\nsummary (per-env marginal cost, from consecutive successful sizes):")
prev = None
for n, tp, tr, drv, base in results:
    if tp is None:
        print(f"  {n:5d}  failed to build")
        continue
    marginal = ""
    if prev is not None and prev[0] is not None:
        dn = n - prev[0]
        if dn > 0 and prev[1] is not None:
            marginal = f"  +{(tp - prev[1]) / dn * 1024:7.2f} MiB/env (torch)"
    print(f"  {n:5d}  torch_peak={tp:6.2f}  driver_total={drv:6.2f}{marginal}")
    prev = (n, tp, tr, drv, base)
