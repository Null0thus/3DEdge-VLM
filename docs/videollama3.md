# VideoLLaMA3 Experiments

This project supports VideoLLaMA3 through `--model-family videollama3`.
The adapter loads the local HuggingFace-style model directory with
`trust_remote_code=True`, then patches the live model instance. It does not edit
files under `models/VideoLLaMA3-7B`.

## Methods

Use `--method` to select the compression policy:

- `full`: keep every VideoLLaMA3 visual token.
- `difffp`: use VideoLLaMA3 Differential Frame Pruning. It supports
  `--videollama3-difffp-selection threshold` for the native threshold rule and
  `--videollama3-difffp-selection topk` for a fixed-budget top-k variant.
- `random`: random pruning on the same post-merge video token grid.
- `ours`: 3D spatio-temporal edge evidence pruning on the same post-merge grid.

`evs` is intentionally not enabled for VideoLLaMA3 until we define a matching
embedding-space baseline for this architecture.

## Important Paths

```bash
--model-family videollama3
--model-path models/VideoLLaMA3-7B
--videollama3-backend hf_local
```

Startup logs print the loaded model class and source file. Check
`outputs/<experiment>/<run>/logs/run.log` to verify the active code path.

## Single-GPU Smoke Test

```bash
cd /data1/data2/csy/exam2

CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py \
  --model-family videollama3 \
  --model-path models/VideoLLaMA3-7B \
  --dataset mvbench \
  --method difffp \
  --num-frames 32 \
  --frame-sampling fps \
  --video-fps 1 \
  --force-sample false \
  --limit-samples 2 \
  --output-dir outputs \
  --experiment-name sanity_videollama3_difffp \
  --run-name chunk0
```

## DiffFP Selection Modes

Native DiffFP keeps tokens whose adjacent-frame difference is above a fixed
threshold. Its actual keep ratio is data-dependent:

```bash
--method difffp \
--videollama3-difffp-selection threshold \
--videollama3-difffp-threshold 0.1 \
--videollama3-difffp-min-tokens 1
```

The top-k variant uses the same adjacent-frame difference score, but fixes the
token budget with `--keep-ratio`. The first frame is kept as the native visual
anchor, and the remaining budget is filled by top-k difference scores:

```bash
--method difffp \
--videollama3-difffp-selection topk \
--keep-ratio 0.20 \
--topk-rounding round \
--videollama3-difffp-min-tokens 1
```

## Five-GPU Full MVBench Example

```bash
cd /data1/data2/csy/exam2
EXP=mvbench_videollama3_ours_keep020_fps1_max180

(
  trap 'jobs -pr | xargs -r kill; wait; exit 130' INT TERM

  for IDX in 0 1 2 3 4; do
    GPU=$((IDX + 3))
    SAVE_HEATMAP=false
    if [ "$IDX" -eq 0 ]; then
      SAVE_HEATMAP=true
    fi
    CUDA_VISIBLE_DEVICES=$GPU python scripts/run_eval.py \
      --model-family videollama3 \
      --model-path models/VideoLLaMA3-7B \
      --dataset mvbench \
      --method ours \
      --keep-ratio 0.20 \
      --sampling-mode bernoulli \
      --window-size 8 \
      --pi-min 0.0667 \
      --pi-max 0.80 \
      --num-frames 180 \
      --frame-sampling fps \
      --video-fps 1 \
      --force-sample false \
      --missing-video-policy skip \
      --save-heatmap $SAVE_HEATMAP \
      --heatmap-save-count 20 \
      --num-chunks 5 \
      --chunk-idx $IDX \
      --chunk-strategy round_robin \
      --output-dir outputs \
      --experiment-name $EXP \
      --run-name chunk$IDX &
  done

  wait
)
```

Merge after all chunks finish:

```bash
EXP=mvbench_videollama3_ours_keep020_fps1_max180

python scripts/merge_runs.py \
  --output-dir outputs/$EXP/merged \
  --run-dirs \
    outputs/$EXP/chunk0 \
    outputs/$EXP/chunk1 \
    outputs/$EXP/chunk2 \
    outputs/$EXP/chunk3 \
    outputs/$EXP/chunk4

cat outputs/$EXP/merged/summary.txt
```

## Output

Each run writes the same files as LLaVA-Video experiments:

```text
args.json
predictions.jsonl
token_stats.jsonl
timings.jsonl
skipped.jsonl
summary.json
summary.txt
logs/run.log
heatmaps/
```

VideoLLaMA3 token grids may be non-square. Heatmaps are saved with the real
post-merge grid shape `(H_p, W_p)` and overlaid on sampled video frames.
