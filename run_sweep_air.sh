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
job_idx=0
#MAX_JOBS_PER_GPU=1
MIN_FREE_MEM=3000

# GPU 메모리 확인 함수
is_gpu_free() {
  local gpu_id=$1
  local free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((gpu_id+1))p")
  if [ "$free_mem" -ge "$MIN_FREE_MEM" ]; then
    return 0 # True (free)
  else
    return 1 # False (busy)
  fi
}

#declare -A GPU_JOB_COUNT

#job_idx=0

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
#SWEEP_PARAMS[16]="cosine 10 5e-5 0.6 linear_residual linear 100 0.01
#cosine 7 1e-4 0.5 linear_residual constant 100 0.01
#cosine 7 1e-4 0.6 linear_residual linear 100 0.01
#cosine 7 1e-5 0.5 linear_residual linear 100 0.01
#cosine 7 1e-5 0.6 linear_residual constant 100 0.01
#cosine 10 5e-5 0.6 linear_residual linear 110 0.005
#cosine 15 1e-4 0.7 self_attn linear 120 0.01
#cosine 15 5e-5 0.7 self_attn constant 120 0.005
#cosine 15 5e-5 0.6 linear constant 120 0.01
#cosine 15 1e-4 0.7 linear_residual linear 120 0.01
#cosine 15 1e-4 0.8 self_attn linear 130 0.005"
#add
SWEEP_PARAMS[16]="cosine 7 5e-5 0.6 linear_residual linear 100 0.01
cosine 7 8e-5 0.6 self_attn linear 100 0.01
cosine 10 1e-5 0.7 linear_residual constant 120 0.05
cosine 10 5e-5 0.7 self_attn constant 120 0.05
cosine 5 8e-5 0.5 linear_residual linear 80 0.01
cosine 5 1e-5 0.5 self_attn linear 80 0.01
cosine 7 5e-5 0.6 linear linear 100 0.05
cosine 10 8e-5 0.7 self_attn linear 120 0.01"




# === Sweep loop ===
for SHOTS in 1 2 4 8 16; do
for SHOTS in 16; do
  while IFS= read -r line; do
    read -r scheduler warmup cons_lr ratio adapter warmup_type max_epoch weight_decay <<< "$line"
    
    # 1. 다음 작업에 대한 GPU를 순차적으로 결정
    target_gpu=${GPUS[$((job_idx % NUM_GPUS))]}
    ((job_idx++))
    
    echo "Waiting for GPU ${target_gpu} to be free..."

    # 2. 결정된 GPU의 메모리가 확보될 때까지 대기
    while ! is_gpu_free $target_gpu; do
      sleep 10
    done
    
    # 3. 작업 실행
    TIME=$(date +%F_%H-%M-%S)
    OUTDIR=output/${DATASET}_grid_search/shots_${SHOTS}/${scheduler}_warm${warmup}_clr${cons_lr}_r${ratio}_${adapter}_wu${warmup_type}_ep${max_epoch}_wd${weight_decay}_${TIME}
    mkdir -p ${OUTDIR}
    
    echo "[GPU $target_gpu STARTING] Shots=${SHOTS}, Scheduler=${scheduler}, Warmup=${warmup}, CLR=${cons_lr}, Ratio=${ratio}, Adapter=${adapter}"
    
    CUDA_VISIBLE_DEVICES=${target_gpu} \
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
      > ${OUTDIR}/log.txt 2>&1 &

    # (선택 사항) 다음 작업 할당 전 약간의 딜레이를 주어 메모리 할당 안정성을 높임
    sleep 5 

  done <<< "${SWEEP_PARAMS[$SHOTS]}"
done

wait
echo "All sweep jobs finished"

