#!/usr/bin/env bash
set -u

# Launch one run_eval.py worker per GPU without letting background workers
# read from or write to the controlling terminal. This avoids bash job-control
# suspensions such as "Stopped" when a worker touches stdin/stdout/stderr.

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="outputs"
EXPERIMENT_NAME=""
GPUS_ARG=""
CHUNK_STRATEGY="round_robin"
RUN_PREFIX="chunk"
SAVE_HEATMAP_FIRST_CHUNK="false"
HEATMAP_SAVE_COUNT=""
RUN_EVAL_ARGS=()
PIDS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/launch_chunked_eval.sh \
    --gpus 0,1,2,3 \
    --experiment-name EXP_NAME \
    [--output-dir outputs] \
    [--chunk-strategy round_robin] \
    [--run-prefix chunk] \
    [--save-heatmap-first-chunk true] \
    [--heatmap-save-count 20] \
    -- RUN_EVAL_ARGS...

Everything after "--" is passed to scripts/run_eval.py.
EOF
}

parse_launcher_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpus)
        GPUS_ARG="$2"
        shift 2
        ;;
      --experiment-name)
        EXPERIMENT_NAME="$2"
        shift 2
        ;;
      --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --chunk-strategy)
        CHUNK_STRATEGY="$2"
        shift 2
        ;;
      --run-prefix)
        RUN_PREFIX="$2"
        shift 2
        ;;
      --save-heatmap-first-chunk)
        SAVE_HEATMAP_FIRST_CHUNK="$2"
        shift 2
        ;;
      --heatmap-save-count)
        HEATMAP_SAVE_COUNT="$2"
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      --)
        shift
        RUN_EVAL_ARGS=("$@")
        break
        ;;
      *)
        echo "Unknown launcher argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done

  if [[ -z "$GPUS_ARG" || -z "$EXPERIMENT_NAME" || ${#RUN_EVAL_ARGS[@]} -eq 0 ]]; then
    usage >&2
    exit 2
  fi
}

cleanup_children() {
  local code="${1:-130}"
  trap - INT TERM
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "Stopping ${#PIDS[@]} chunk workers..."
    for pid in "${PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  fi
  exit "$code"
}

parse_launcher_args "$@"

IFS=',' read -r -a GPUS <<< "$GPUS_ARG"
NUM_CHUNKS="${#GPUS[@]}"
if [[ "$NUM_CHUNKS" -le 0 ]]; then
  echo "No GPUs were provided." >&2
  exit 2
fi

trap 'cleanup_children 130' INT TERM

for IDX in "${!GPUS[@]}"; do
  GPU="${GPUS[$IDX]}"
  RUN_NAME="${RUN_PREFIX}${IDX}"
  RUN_DIR="${OUTPUT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}"
  LOG_DIR="${RUN_DIR}/logs"
  mkdir -p "$LOG_DIR"

  SAVE_HEATMAP="false"
  if [[ "$SAVE_HEATMAP_FIRST_CHUNK" == "true" && "$IDX" -eq 0 ]]; then
    SAVE_HEATMAP="true"
  fi

  CMD=(
    "$PYTHON_BIN" scripts/run_eval.py
    "${RUN_EVAL_ARGS[@]}"
    --output-dir "$OUTPUT_DIR"
    --experiment-name "$EXPERIMENT_NAME"
    --run-name "$RUN_NAME"
    --num-chunks "$NUM_CHUNKS"
    --chunk-idx "$IDX"
    --chunk-strategy "$CHUNK_STRATEGY"
    --save-heatmap "$SAVE_HEATMAP"
  )
  if [[ -n "$HEATMAP_SAVE_COUNT" ]]; then
    CMD+=(--heatmap-save-count "$HEATMAP_SAVE_COUNT")
  fi

  echo "Starting ${RUN_NAME} on GPU ${GPU}; log: ${LOG_DIR}/terminal.log"
  CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" < /dev/null > "${LOG_DIR}/terminal.log" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    STATUS=1
  fi
done

if [[ "$STATUS" -ne 0 ]]; then
  echo "At least one chunk failed. Check ${OUTPUT_DIR}/${EXPERIMENT_NAME}/chunk*/logs/terminal.log" >&2
  exit "$STATUS"
fi

echo "All chunks finished for ${EXPERIMENT_NAME}."
