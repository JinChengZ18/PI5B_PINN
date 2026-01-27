#!/bin/bash

# ================================
# 可用 GPU 列表（GPU 1–5）
# ================================
GPUS=(1 2 3 4 5)
NUM_GPUS=${#GPUS[@]}

# ================================
# 所有实验参数列表（可扩展到任意数量）
# ================================
EXPERIMENTS=(
  "--epochs 80 --lr 1e-4   --points_per_case 4096 --log_dir logs/M01/exp01 --ckpt_dir checkpoints/M01/exp01"
  "--epochs 80 --lr 5e-4   --points_per_case 6144 --log_dir logs/M01/exp02 --ckpt_dir checkpoints/M01/exp02"
  "--epochs 80 --lr 1e-3   --points_per_case 8192 --log_dir logs/M01/exp03 --ckpt_dir checkpoints/M01/exp03"
  "--epochs 100 --lr 1e-4   --points_per_case 6144 --log_dir logs/M01/exp04 --ckpt_dir checkpoints/M01/exp04"
  "--epochs 100 --lr 5e-4   --points_per_case 8192 --log_dir logs/M01/exp05 --ckpt_dir checkpoints/M01/exp05"
  "--epochs 100 --lr 1e-3   --points_per_case 4096 --log_dir logs/M01/exp06 --ckpt_dir checkpoints/M01/exp06"
  "--epochs 120 --lr 1e-4   --points_per_case 8192 --log_dir logs/M01/exp07 --ckpt_dir checkpoints/M01/exp07"
  "--epochs 120 --lr 5e-4   --points_per_case 4096 --log_dir logs/M01/exp08 --ckpt_dir checkpoints/M01/exp08"
  "--epochs 120 --lr 1e-3   --points_per_case 6144 --log_dir logs/M01/exp09 --ckpt_dir checkpoints/M01/exp09"
)

# ================================
# 启动任务（轮转 GPU 分配）
# ================================
for i in "${!EXPERIMENTS[@]}"; do
  GPU_INDEX=$((i % NUM_GPUS))
  GPU_ID=${GPUS[$GPU_INDEX]}

  echo "Launching experiment $((i+1)) on GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES=$GPU_ID \
  python train.py \
    --num_workers 8 \
    ${EXPERIMENTS[$i]} &

  # 控制并发数量：最多同时运行 NUM_GPUS 个任务
  if (( (i+1) % NUM_GPUS == 0 )); then
    wait
  fi
done

# 等待最后一批任务完成
wait
echo "✅ 所有任务已完成"

