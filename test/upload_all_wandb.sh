#!/usr/bin/env bash

# 离线 wandb 日志目录（请根据实际情况修改）
WANDB_DIR="/volume/pt-train/users/mingjie/hzl_code/code/slime/experiments/record/2task-test/wandb"

# 写死的排除文件/文件夹列表
EXCLUDE_FILES=("latest-run" "debug-internal.log" "debug.log")

# 检查 wandb 是否安装
if ! command -v wandb &> /dev/null; then
    echo "wandb 命令未找到，请先安装 wandb: pip install wandb"
    exit 1
fi

if [ ! -d "$WANDB_DIR" ]; then
    echo "指定的目录不存在: $WANDB_DIR"
    exit 1
fi

# 遍历主目录下的所有子文件夹
for subdir in "$WANDB_DIR"/*; do
    # 跳过不是目录的
    [ -d "$subdir" ] || continue

    # 检查是否在排除列表
    skip=0
    for ex in "${EXCLUDE_FILES[@]}"; do
        if [[ "$(basename "$subdir")" == "$ex" ]]; then
            skip=1
            break
        fi
    done
    [ $skip -eq 1 ] && continue

    echo "正在上传: $subdir"
    wandb sync "$subdir"
done