# HSI（Perceptive Shadowing / Perceptive Track）

本目录为 `instinctlab.tasks.HSI` 任务包（跟踪与环境实现在 `perceptive_track/`）：基于 perceptive shadowing，使用独立 Gym 环境 ID，与 `tasks/shadowing/perceptive` 可并存。

## Perceptive Shadowing（PPO）

**Task ID:** `Instinct-HSITrack-Perceptive-Shadowing-G1-v0`

1. 在 `perceptive_track/config/g1/perceptive_shadowing_cfg.py` 中设置 `MOTION_FOLDER` 及运动数据路径；`motion_buffer` 与地形生成器会读取对应目录及 `metadata.yaml`。

2. 训练：

```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-HSITrack-Perceptive-Shadowing-G1-v0
```

3. 回放（需 `--load_run` 或 `--no_resume`）：

```bash
python source/instinctlab/instinctlab/tasks/HSI/play.py --task=Instinct-HSITrack-Perceptive-Shadowing-G1-Play-v0 --load_run=<run_name>
```

## Perceptive VAE

**Task IDs:** `Instinct-HSITrack-Perceptive-Vae-G1-v0` / `Instinct-HSITrack-Perceptive-Vae-G1-Play-v0`

训练与回放时将 `--task=` 改为上述名称即可；回放脚本路径同上 `tasks/HSI/play.py`。

## Common Options

- `--num_envs`: 并行环境数
- `--max_iterations`: 训练迭代次数
- `--load_run`: 加载的检查点 run 目录名
- `--video`: 录制视频
