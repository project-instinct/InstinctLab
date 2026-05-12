# Parkour Task

## Basic Usage Guidelines

### Parkour Task

**Task ID:** `Instinct-Parkour-Target-Amp-G1-v0`

1. Edit `<repository-root>/data/dataset_folder/parkour.yaml`: list reference motions under `selected_files` with paths **relative to** `data/dataset_folder` (same convention as BeyondMimic’s `beyondmimic.yaml`). To change the motion root or YAML filename instead, edit `AmassMotionCfg.path` and `filtered_motion_selection_filepath` in `config/g1/g1_parkour_target_amp_cfg.py`.

2. Train the policy:
```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name>
```

4. Export trained policy (load_run must be provided, absolute path is recommended):

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name> --exportonnx --useonnx
```

## Common Options

- `--num_envs`: Number of parallel environments (default varies by task)
- `--keyboard_control`: Enable keyboard control during playing
- `--load_run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--exportonnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--useonnx`: Use the ONNX model for inference during playing (requires `--exportonnx`)
