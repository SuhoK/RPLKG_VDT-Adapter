#!/bin/bash

# === Editable Configurations ===
DATA="/home/shkang/VDT_Project/data/fgvc_aircraft"
CFG="vit_l14"
TRAINER="CLIP_Adapter_gpt"
DATASET="fgvc_aircraft"
SEED=1
TOPK=5

# GPU list
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
MAX_JOBS_PER_GPU=1

declare -A GPU_JOB_COUNT
for gpu_id in "${GPUS[@]}"; do
    GPU_JOB_COUNT[$gpu_id]=0
done

job_idx=0

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

    # 사용 가능한 GPU가 생길 때까지 기다리기
    while true; do
      for gpu_id in "${GPUS[@]}"; do
        # 현재 GPU에 job이 MAX_JOBS_PER_GPU 이하인 경우 선택
        if [ ${GPU_JOB_COUNT[$gpu_id]} -lt $MAX_JOBS_PER_GPU ]; then
          GPU_JOB_COUNT[$gpu_id]=$((GPU_JOB_COUNT[$gpu_id]+1))

          TIME=$(date +%F_%H-%M-%S)
          OUTDIR=output/${DATASET}_grid_search/shots_${SHOTS}/${scheduler}_warm${warmup}_clr${cons_lr}_r${ratio}_${adapter}_wu${warmup_type}_ep${max_epoch}_wd${weight_decay}_${TIME}
          mkdir -p ${OUTDIR}

          echo "[GPU $gpu_id STARTING] Shots=${SHOTS}, Scheduler=${scheduler}, Warmup=${warmup}, CLR=${cons_lr}, Ratio=${ratio}, Adapter=${adapter}"

          (
          CUDA_VISIBLE_DEVICES=${gpu_id} \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          python train.py \
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
            > ${OUTDIR}/log.txt 2>&1

          # 종료되면 job 수 감소
          GPU_JOB_COUNT[$gpu_id]=$((GPU_JOB_COUNT[$gpu_id]-1))
          ) &

          # 다음 sweep 줄로 이동
          break 2
        fi
      done

      # GPU가 모두 바쁘면 10초 기다리기
      sleep 10
    done

  done <<< "${SWEEP_PARAMS[$SHOTS]}"
done

wait
echo "All sweep jobs finished"
