#!/bin/bash

DATA="your/data/path"
CFG="vit_l14"
TRAINER="CLIP_Adapter_gpt"
DATASET="dtd"
SEED=1

# GPU list
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
job_idx=0

# Define parameter arrays for each shot condition
declare -A LR_SCHEDULER
declare -A WARMUP_EPOCH
declare -A WARMUP_CONS_LR
declare -A RATIO
declare -A WORD_ADAPTER_TYPE

# Recommended sweep settings per shot
LR_SCHEDULER[1]="cosine"
LR_SCHEDULER[2]="cosine"
LR_SCHEDULER[4]="cosine single_step"
LR_SCHEDULER[8]="single_step cosine"
LR_SCHEDULER[16]="single_step cosine"

WARMUP_EPOCH[1]="3 5"
WARMUP_EPOCH[2]="3 5"
WARMUP_EPOCH[4]="5 7"
WARMUP_EPOCH[8]="7 10"
WARMUP_EPOCH[16]="7 10 15"

WARMUP_CONS_LR[1]="1e-4"
WARMUP_CONS_LR[2]="1e-4"
WARMUP_CONS_LR[4]="1e-4 1e-5"
WARMUP_CONS_LR[8]="1e-4 1e-5"
WARMUP_CONS_LR[16]="1e-4 1e-5"

RATIO[1]="0.2"
RATIO[2]="0.2"
RATIO[4]="0.2 0.5"
RATIO[8]="0.2 0.5"
RATIO[16]="0.2 0.5"

WORD_ADAPTER_TYPE[1]="linear"
WORD_ADAPTER_TYPE[2]="linear linear_residual"
WORD_ADAPTER_TYPE[4]="linear linear_residual"
WORD_ADAPTER_TYPE[8]="linear_residual self_attn"
WORD_ADAPTER_TYPE[16]="linear_residual self_attn"

# Sweep loop
for SHOTS in 1 2 4 8 16
do
  for scheduler in ${LR_SCHEDULER[$SHOTS]}
  do
    for warmup in ${WARMUP_EPOCH[$SHOTS]}
    do
      for cons_lr in ${WARMUP_CONS_LR[$SHOTS]}
      do
        for ratio in ${RATIO[$SHOTS]}
        do
          for adapter in ${WORD_ADAPTER_TYPE[$SHOTS]}
          do
            # Round-robin GPU assignment
            gpu_id=${GPUS[$((job_idx % NUM_GPUS))]}

            TIME=$(date +%F_%H-%M-%S)
            OUTDIR=output/DTD_grid_search/shots_${SHOTS}/${scheduler}_warm${warmup}_clr${cons_lr}_r${ratio}_${adapter}_${TIME}
            mkdir -p ${OUTDIR}
            
            echo "[GPU $gpu_id STARTING] Shots=${SHOTS}, Scheduler=${scheduler}, Warmup=${warmup}, CLR=${cons_lr}, Ratio=${ratio}, Adapter=${adapter}"

            CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
              --root ${DATA} \
              --seed ${SEED} \
              --trainer ${TRAINER} \
              --dataset-config-file configs/datasets/${DATASET}.yaml \
              --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
              --output-dir ${OUTDIR} \
              OPTIM.LR_SCHEDULER ${scheduler} \
              OPTIM.WARMUP_EPOCH ${warmup} \
              OPTIM.WARMUP_CONS_LR ${cons_lr} \
              TRAINER.CLIP_ADAPTER.RATIO ${ratio} \
              TRAINER.CLIP_ADAPTER.WORD_ADAPTER_TYPE ${adapter} \
              DATASET.NUM_SHOTS ${SHOTS} \
              > ${OUTDIR}/log.txt 2>&1 & # Run in background

            # Increase job index
            ((job_idx++))

            # Optional: Limit parallel jobs (uncomment if needed)
            # sleep 1

          done
        done
      done
    done
  done
done

wait # Wait until all background jobs complete
echo "All sweep jobs finished successfully"
