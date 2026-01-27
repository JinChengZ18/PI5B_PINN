#!/bin/bash

# ================================
# GPU 与并发配置
# ================================
GPUS=(1 2 3 4 5)
TOTAL_GPUS=${#GPUS[@]}

MAX_TASKS_PER_GPU=2
MAX_CONCURRENT=$((TOTAL_GPUS * MAX_TASKS_PER_GPU))

echo "Using GPUs: ${GPUS[@]}"
echo "Max concurrent jobs: ${MAX_CONCURRENT}"

# ================================
# 实验参数
# ================================
EXPERIMENTS=(
  "--epochs 120 --lr 5e-3 --points_per_case 10000 --log_dir logs/M02/exp01 --ckpt_dir checkpoints/M02/exp01"
  "--epochs 120 --lr 5e-3 --points_per_case 20000 --log_dir logs/M02/exp02 --ckpt_dir checkpoints/M02/exp02"
  "--epochs 120 --lr 5e-3 --points_per_case 40000 --log_dir logs/M02/exp03 --ckpt_dir checkpoints/M02/exp03"
  "--epochs 120 --lr 5e-3 --points_per_case 80000 --log_dir logs/M02/exp04 --ckpt_dir checkpoints/M02/exp04"
  "--epochs 120 --lr 5e-3 --points_per_case 100000 --log_dir logs/M02/exp05 --ckpt_dir checkpoints/M02/exp05"
  "--epochs 120 --lr 5e-3 --points_per_case 200000 --log_dir logs/M02/exp06 --ckpt_dir checkpoints/M02/exp06"
  "--epochs 120 --lr 5e-3 --points_per_case 300000 --log_dir logs/M02/exp07 --ckpt_dir checkpoints/M02/exp07"
  "--epochs 120 --lr 5e-3 --points_per_case 400000 --log_dir logs/M02/exp08 --ckpt_dir checkpoints/M02/exp08"
  "--epochs 120 --lr 5e-3 --points_per_case 600000 --log_dir logs/M02/exp09 --ckpt_dir checkpoints/M02/exp09"
)

# ================================
# 启动任务（防断线 + 并发控制）
# ================================
for i in "${!EXPERIMENTS[@]}"; do
  GPU_INDEX=$((i % TOTAL_GPUS))
  GPU_ID=${GPUS[$GPU_INDEX]}

  # ---- 并发限制 ----
  while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENT" ]; do
    sleep 1
  done

  EXP_ID=$(printf "%02d" $((i+1)))
  LOG_DIR="logs/exp${EXP_ID}"
  mkdir -p "$LOG_DIR"

  echo "Launching experiment ${EXP_ID} on GPU ${GPU_ID}"

  nohup bash -c "
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python train.py \
      --num_workers 8 \
      ${EXPERIMENTS[$i]}
  " > "${LOG_DIR}/stdout.log" 2>&1 &

  disown
done

echo "🚀 All jobs submitted safely (SSH-safe)"
