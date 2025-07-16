#!/bin/bash

# === Editable Configurations ===
DATA="/home/shkang/VDT_Project/data/dtd"
CFG="vit_l14"
TRAINER="CLIP_Adapter_gpt"
DATASET="dtd"
SEED=1

# GPU list
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

gpu_id=${GPUS[$((job_idx % NUM_GPUS))]}

# Define parameter combinations per shot
declare -A SWEEP_PARAMS

# 1 shot (2 combinations)
SWEEP_PARAMS[1]="cosine 3 5e-5 0.5 linear_residual
cosine 5 1e-4 0.3 linear_residual"


# 2 shot (4 combinations)
SWEEP_PARAMS[2]="cosine 5 5e-5 0.5 linear_residual
cosine 7 1e-4 0.4 self_attn
cosine 5 1e-4 0.6 linear_residual"


# 4 shot (8 combinations)
SWEEP_PARAMS[4]="cosine 7 1e-4 0.5 linear_residual
cosine 10 1e-4 0.6 self_attn
cosine 7 5e-5 0.4 linear_residual
cosine 7 1e-4 0.7 self_attn"


# 8 shot (8 combinations)
SWEEP_PARAMS[8]="cosine 7 1e-4 0.5 linear_residual
cosine 10 1e-4 0.6 linear_residual
cosine 10 5e-5 0.7 self_attn
cosine 7 1e-4 0.6 self_attn"

# 16 shot (16 combinations)
SWEEP_PARAMS[16]="cosine 10 1e-4 0.5 linear_residual
cosine 15 1e-4 0.6 self_attn
cosine 15 5e-5 0.7 linear_residual
cosine 12 1e-4 0.6 self_attn
cosine 10 1e-4 0.4 linear_residual"


# === Sweep loop ===
for SHOTS in 1 2 4 8 16
do
  while IFS= read -r line; do
    read -r scheduler warmup cons_lr ratio adapter <<< "$line"

    gpu_id=${GPUS[$((job_idx % NUM_GPUS))]}
    TIME=$(date +%F_%H-%M-%S)
    OUTDIR=output/${DATASET}_grid_search/shots_${SHOTS}/${scheduler}_warm${warmup}_clr${cons_lr}_r${ratio}_${adapter}_${TIME}
    mkdir -p ${OUTDIR}

    echo "[GPU $gpu_id STARTING] Shots=${SHOTS}, Scheduler=${scheduler}, Warmup=${warmup}, CLR=${cons_lr}, Ratio=${ratio}, Adapter=${adapter}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${OUTDIR} \
      --optim adam \
      OPTIM.LR_SCHEDULER ${scheduler} \
      OPTIM.WARMUP_EPOCH ${warmup} \
      OPTIM.WARMUP_CONS_LR ${cons_lr} \
      TRAINER.CLIP_ADAPTER.RATIO ${ratio} \
      TRAINER.CLIP_ADAPTER.WORD_ADAPTER_TYPE ${adapter} \
      DATASET.NUM_SHOTS ${SHOTS} \
      > ${OUTDIR}/log.txt 2>&1 &

    ((job_idx++))
  done <<< "${SWEEP_PARAMS[$SHOTS]}"
done

wait
echo "All sweep jobs finished"
