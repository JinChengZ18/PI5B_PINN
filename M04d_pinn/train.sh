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
  # ===== λ_pde 敏感性（S1–S5）=====
  "--lambda_pde 0.00 --epochs 200 --lr 5e-3 --use_pde_anneal \
   --log_dir logs/M04d/exp01 --ckpt_dir checkpoints/M04d/exp01"

  "--lambda_pde 0.25 --epochs 200 --lr 5e-3 --use_pde_anneal \
   --log_dir logs/M04d/exp02 --ckpt_dir checkpoints/M04d/exp02"

  "--lambda_pde 0.50 --epochs 200 --lr 5e-3 --use_pde_anneal \
   --log_dir logs/M04d/exp03 --ckpt_dir checkpoints/M04d/exp03"

  "--lambda_pde 2.00 --epochs 200 --lr 5e-3 --use_pde_anneal \
   --log_dir logs/M04d/exp04 --ckpt_dir checkpoints/M04d/exp04"

  "--lambda_pde 3.00 --epochs 200 --lr 5e-3 --use_pde_anneal \
   --log_dir logs/M04d/exp05 --ckpt_dir checkpoints/M04d/exp05"


  # ===== Annealing γ 敏感性（A1–A5）=====
  "--lambda_pde 1.0 --epochs 200 --lr 5e-3 \
   --use_pde_anneal --pde_anneal_gamma 0.10 \
   --log_dir logs/M04d/exp06 --ckpt_dir checkpoints/M04d/exp06"

  "--lambda_pde 1.0 --epochs 200 --lr 5e-3 \
   --use_pde_anneal --pde_anneal_gamma 0.50 \
   --log_dir logs/M04d/exp07 --ckpt_dir checkpoints/M04d/exp07"

  "--lambda_pde 1.0 --epochs 200 --lr 5e-3 \
   --use_pde_anneal --pde_anneal_gamma 1.00 \
   --log_dir logs/M04d/exp08 --ckpt_dir checkpoints/M04d/exp08"

  "--lambda_pde 1.0 --epochs 200 --lr 5e-3 \
   --use_pde_anneal --pde_anneal_gamma 2.00 \
   --log_dir logs/M04d/exp08 --ckpt_dir checkpoints/M04d/exp09"

  "--lambda_pde 1.0 --epochs 200 --lr 5e-3 \
   --use_pde_anneal --pde_anneal_gamma 3.00 \
   --log_dir logs/M04d/exp08 --ckpt_dir checkpoints/M04d/exp09"
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
