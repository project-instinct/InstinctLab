#!/bin/bash

# 1. 既然你在终端已经 activate 了环境，脚本里不需要这行，否则会报错
# conda activate isaaclab

# 2. 定义变量
PYTHON_SCRIPT="source/instinctlab/instinctlab/tasks/HSI/play.py"
TASK_NAME="Instinct-HSITrack-Perceptive-Shadowing-G1-Play-v0"
LOAD_RUN=".*034830.*"
NUM_ENVS=100
X_VALS=$(seq -0.6 0.1 0.6)
Y_VALS=$(seq -0.6 0.1 0.6)

# 3. 循环运行
for x in $X_VALS; do
    for y in $Y_VALS; do
        echo "=================================================="
        echo "Running Grid Search: x=$x, y=$y"
        echo "=================================================="

        # 调用 Python 脚本
        # 注意：使用 python (也就是当前 conda 环境的 python)
        python "$PYTHON_SCRIPT" \
            --task "$TASK_NAME" \
            --load_run "$LOAD_RUN" \
            --num_envs "$NUM_ENVS" \
            --x_offset "$x" \
            --y_offset "$y" \
            --headless

        # 检查上一步是否成功
        if [ $? -ne 0 ]; then
            echo "[ERROR] Simulation failed for x=$x, y=$y. Continuing..."
        fi

        # 稍微暂停一下，确保端口释放
        sleep 20
    done
done

echo "Grid Search All Finished!"
