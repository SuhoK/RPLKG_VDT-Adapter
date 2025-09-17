#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ==============================
# 기본 설정 (환경변수로 덮어쓰기 가능)
# ==============================
DATA="${DATA:-/data1/suho/DATA}"                  # 데이터 루트
DATASET="${DATASET:-oxford_pets}"                 # DATASET=flowers 등으로 덮기
SEEDS=(${SEEDS:-1 2 3})                           # 시드 세트
GPUS=(${GPUS:-0 1 2 3})                           # 사용 GPU 목록
MIN_FREE_MEM="${MIN_FREE_MEM:-20000}"             # MiB (메모리 넉넉히 필터링)
PYTHON="${PYTHON:-python}"

TRAINER="${TRAINER:-CLIP_Adapter_gpt}"            # VDT-Adapter 트레이너
CFG="${CFG:-vit_b16_c16_ep10_batch1}"             # ViT-B/16 구성 파일명

OUT_ROOT="output/${DATASET}_vdt_adapter_vitb16_k1_2_4_8_16"

# =====================================================
# GPU 선택: free mem 큰 카드 선택 (MIN_FREE_MEM 이상만)
# =====================================================
choose_gpu() {
  mapfile -t frees < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  local best=-1 best_free=-1
  for id in "${GPUS[@]}"; do
    local free="${frees[$id]:-0}"
    if [[ "$free" -ge "$MIN_FREE_MEM" && "$free" -gt "$best_free" ]]; then
      best=$id; best_free=$free
    fi
  done
  echo "$best"
}

# ===== 스윕 스펙: K2_SGD + K2_ADAMW를 최대한 유지 =====
# 포맷: OPT|LR|WD|EPOCH|N_CTX|CSC|CTP|SCHED|WARMUP|WU_TYPE|CONS_LR|RATIO|ADAPTER
SPECS=(
  # Pair 1 (ep90, wu5, ratio0.3, linear_residual)
  "adamw|1e-4|0.01|90|16|False|end|cosine|5|linear|0|0.3|linear_residual"
  "sgd|1e-3|5e-4|90|16|False|end|cosine|5|linear|0|0.3|linear_residual"

  # Pair 2 (ep100, wu7, ratio0.4, self_attn)
  "adamw|2e-4|0.01|100|16|False|end|cosine|7|linear|0|0.4|self_attn"
  "sgd|2e-3|1e-3|100|16|False|end|cosine|7|linear|0|0.4|self_attn"

  # Pair 3 (ep100, wu10, ratio0.5, linear, constant warmup)
  "adamw|1e-4|0.005|100|16|False|end|cosine|10|constant|5e-5|0.5|linear"
  "sgd|3e-3|1e-3|100|16|False|end|cosine|10|constant|1e-4|0.5|linear"

  # Pair 4 (ep110, wu10, ratio0.5, linear_residual)
  "adamw|1.5e-4|0.01|110|16|False|end|cosine|10|linear|0|0.5|linear_residual"
  "sgd|5e-3|1e-3|110|16|False|end|cosine|10|linear|0|0.5|linear_residual"

  # Pair 5 (ep110, wu12, ratio0.6, self_attn)
  "adamw|2e-4|0.005|110|16|False|end|cosine|12|linear|0|0.6|self_attn"
  "sgd|3e-3|5e-4|110|16|False|end|cosine|12|linear|0|0.6|self_attn"

  # Pair 6 (ep120, wu12, ratio0.6, linear_residual)
  "adamw|2.5e-4|0.01|120|16|False|end|cosine|12|linear|0|0.6|linear_residual"
  "sgd|2e-3|5e-4|120|16|False|end|cosine|12|linear|0|0.6|linear_residual"

  # Pair 7 (ep120, wu15, ratio0.7, self_attn)
  "adamw|3e-4|0.005|120|16|False|end|cosine|15|linear|0|0.7|self_attn"
  "sgd|5e-3|1e-3|120|16|False|end|cosine|15|linear|0|0.7|self_attn"

  # Pair 8 (ep80, wu5, ratio0.3, linear)
  "adamw|1.5e-4|0.01|80|16|False|end|cosine|5|linear|0|0.3|linear"
  "sgd|2e-3|5e-4|80|16|False|end|cosine|5|linear|0|0.3|linear"

  # Pair 9 (ep90, wu7, ratio0.5, linear_residual, constant warmup)
  "adamw|2e-4|0.01|90|16|False|end|cosine|7|constant|5e-5|0.5|linear_residual"
  "sgd|3e-3|1e-3|90|16|False|end|cosine|7|constant|1e-4|0.5|linear_residual"

  # Pair 10 (ep100, wu7, ratio0.6, self_attn)
  "adamw|1e-4|0.005|100|16|False|end|cosine|7|linear|0|0.6|self_attn"
  "sgd|1e-3|5e-4|100|16|False|end|cosine|7|linear|0|0.6|self_attn"
)


run_one() {
  local SHOTS=$1 OPT=$2 LR=$3 WD=$4 EPOCH=$5 NCTX=$6 CSC=$7 CTP=$8 \
        SCHED=$9 WARMUP=${10} WU_TYPE=${11} CONS_LR=${12} \
        RATIO=${13} ADAPTER=${14} SEED=${15} GPU=${16}

  local TAG="k${SHOTS}_${OPT}_lr${LR}_wd${WD}_ep${EPOCH}_n${NCTX}_csc${CSC}_ctp${CTP}_${SCHED}_wu${WARMUP}_${WU_TYPE}_cl${CONS_LR}_ratio${RATIO}_${ADAPTER}_seed${SEED}"
  local OUTDIR="${OUT_ROOT}/shots_${SHOTS}/${TAG}"
  mkdir -p "${OUTDIR}"

  # --- WARMUP_CONS_LR 타입/전달 제어 ---
  # 1) constant일 때만 전달
  # 2) float로 맞춤 (0 -> 0.0)
  local CONS_LR_ARG=()
  if [[ "${WU_TYPE}" == "constant" ]]; then
    [[ "${CONS_LR}" == "0" ]] && CONS_LR="0.0"
    CONS_LR_ARG=( OPTIM.WARMUP_CONS_LR "${CONS_LR}" )
  fi

  echo "[LAUNCH][GPU ${GPU}] ${DATASET} ${TAG}"

  CUDA_VISIBLE_DEVICES=${GPU} \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${PYTHON:-python} train.py \
    --root "${DATA}" \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${OUTDIR}" \
    OPTIM.NAME ${OPT} \
    OPTIM.LR ${LR} \
    OPTIM.WEIGHT_DECAY ${WD} \
    OPTIM.MAX_EPOCH ${EPOCH} \
    OPTIM.LR_SCHEDULER ${SCHED} \
    OPTIM.WARMUP_EPOCH ${WARMUP} \
    OPTIM.WARMUP_TYPE ${WU_TYPE} \
    TRAINER.CLIP_ADAPTER.RATIO ${RATIO} \
    TRAINER.CLIP_ADAPTER.WORD_ADAPTER_TYPE ${ADAPTER} \
    DATASET.NUM_SHOTS ${SHOTS} \
    "${CONS_LR_ARG[@]}" \
    > "${OUTDIR}/log.txt" 2>&1 &
}



mkdir -p "${OUT_ROOT}"

# ==============================
# 실행: k=1,2,4,8,16 동일 조건 스윕
# ==============================
for SHOTS in 1 2 4 8 16; do
  for spec in "${SPECS[@]}"; do
    IFS="|" read -r OPT LR WD EPOCH NCTX CSC CTP SCHED WARMUP WU_TYPE CONS_LR RATIO ADAPTER <<< "${spec}"
    for SEED in "${SEEDS[@]}"; do
      gpu=$(choose_gpu)
      while [[ "$gpu" == "-1" ]]; do
        echo "[WAIT] all GPUs < ${MIN_FREE_MEM} MiB free; retry in 10s..."
        sleep 10
        gpu=$(choose_gpu)
      done
      run_one "$SHOTS" "$OPT" "$LR" "$WD" "$EPOCH" "$NCTX" "$CSC" "$CTP" \
              "$SCHED" "$WARMUP" "$WU_TYPE" "$CONS_LR" "$RATIO" "$ADAPTER" \
              "$SEED" "$gpu"
      sleep 2
    done
  done
done


wait
echo "[DONE] VDT-Adapter sweep for ${DATASET}"
echo "[HINT] 결과 모아보기:"
echo "  grep -R \"top1\" ${OUT_ROOT} | sort -V | tail -n 30"
