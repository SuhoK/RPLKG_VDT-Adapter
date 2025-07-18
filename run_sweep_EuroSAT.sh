#!/bin/bash

# === Editable Configurations ===
DATA="/home/shkang/VDT_Project/data/eurosat"
CFG="vit_l14"
TRAINER="CLIP_Adapter_gpt"
DATASET="eurosat"
SEED=1
TOPK=5

# GPU list
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

gpu_id=${GPUS[$((job_idx % NUM_GPUS))]}

# Define parameter combinations per shot
declare -A SWEEP_PARAMS

# === 1-shot ===
SWEEP_PARAMS[1]="cosine 3 1e-5 0.2 linear linear 70 0.01
cosine 5 1e-5 0.3 linear_residual linear 80 0.01
cosine 5 5e-5 0.3 self_attn linear 90 0.005
cosine 7 1e-4 0.3 linear_residual constant 90 0.01
cosine 7 5e-5 0.4 self_attn linear 100 0.01"

# === 2-shot ===
SWEEP_PARAMS[2]="cosine 5 1e-5 0.3 linear linear 70 0.01
cosine 5 5e-5 0.3 self_attn constant 80 0.005
cosine 7 5e-5 0.4 linear_residual linear 90 0.01
cosine 7 1e-4 0.4 linear_residual linear 100 0.01
cosine 10 1e-4 0.5 self_attn linear 100 0.01
cosine 10 5e-5 0.5 linear constant 100 0.005"

# === 4-shot ===
SWEEP_PARAMS[4]="cosine 5 1e-4 0.4 linear linear 90 0.01
cosine 7 1e-4 0.5 linear constant 100 0.01
cosine 7 1e-4 0.5 linear_residual linear 100 0.005
cosine 7 5e-5 0.5 self_attn linear 100 0.01
cosine 10 1e-4 0.4 self_attn constant 110 0.01
cosine 10 5e-5 0.5 linear_residual linear 110 0.005
cosine 10 1e-4 0.6 linear_residual linear 110 0.01"

# === 8-shot ===
SWEEP_PARAMS[8]="cosine 7 1e-4 0.5 linear linear 100 0.01
cosine 7 1e-4 0.5 linear_residual linear 100 0.01
cosine 7 1e-4 0.6 linear_residual constant 100 0.005
cosine 7 1e-5 0.5 linear_residual constant 100 0.01
cosine 7 1e-5 0.6 linear_residual linear 100 0.01
cosine 10 1e-4 0.5 linear_residual linear 110 0.005
cosine 10 5e-5 0.6 self_attn linear 110 0.01
cosine 10 1e-4 0.6 self_attn constant 110 0.005
cosine 12 5e-5 0.6 self_attn linear 120 0.01
cosine 12 1e-4 0.6 linear_residual linear 120 0.005"

# === 16-shot ===
SWEEP_PARAMS[16]="cosine 10 5e-5 0.6 linear_residual linear 100 0.01
cosine 7 1e-4 0.5 linear_residual constant 100 0.01
cosine 7 1e-4 0.6 linear_residual linear 100 0.01
cosine 7 1e-5 0.5 linear_residual linear 100 0.01
cosine 7 1e-5 0.6 linear_residual constant 100 0.01
cosine 10 5e-5 0.6 linear_residual linear 110 0.005
cosine 15 1e-4 0.7 self_attn linear 120 0.01
cosine 15 5e-5 0.7 self_attn constant 120 0.005
cosine 15 5e-5 0.6 linear constant 120 0.01
cosine 15 1e-4 0.7 linear_residual linear 120 0.01
cosine 15 1e-4 0.8 self_attn linear 130 0.005"





# === Sweep loop ===
for SHOTS in 1 2 4 8 16
do
  while IFS= read -r line; do
    read -r scheduler warmup cons_lr ratio adapter warmup_type max_epoch weight_decay <<< "$line"

    gpu_id=${GPUS[$((job_idx % NUM_GPUS))]}
    TIME=$(date +%F_%H-%M-%S)
    OUTDIR=output/${DATASET}_grid_search/shots_${SHOTS}/${scheduler}_warm${warmup}_clr${cons_lr}_r${ratio}_${adapter}_wu${warmup_type}_ep${max_epoch}_wd${weight_decay}_${TIME}
    mkdir -p ${OUTDIR}

    echo "[GPU $gpu_id STARTING] Shots=${SHOTS}, Scheduler=${scheduler}, Warmup=${warmup}, CLR=${cons_lr}, Ratio=${ratio}, Adapter=${adapter}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${OUTDIR} \
      OPTIM.NAME adam \
      OPTIM.LR_SCHEDULER ${scheduler} \
      OPTIM.WARMUP_EPOCH ${warmup} \
      OPTIM.WARMUP_CONS_LR ${cons_lr} \
      OPTIM.WARMUP_TYPE ${warmup_type} \
      OPTIM.MAX_EPOCH ${max_epoch} \
      OPTIM.WEIGHT_DECAY ${weight_decay} \
      TRAINER.CLIP_ADAPTER.RATIO ${ratio} \
      TRAINER.CLIP_ADAPTER.WORD_ADAPTER_TYPE ${adapter} \
      DATASET.NUM_SHOTS ${SHOTS} \
      > ${OUTDIR}/log.txt 2>&1 &

    ((job_idx++))
  done <<< "${SWEEP_PARAMS[$SHOTS]}"
done

wait
echo "All sweep jobs finished"
