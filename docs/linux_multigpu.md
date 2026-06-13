# Linux Multi-GPU Evaluation

Use `scripts/launch_chunked_eval.sh` to run one `run_eval.py` worker per GPU.
The launcher redirects each background worker away from the current terminal:

- stdin: `/dev/null`
- stdout/stderr: `outputs/<experiment>/<chunk>/logs/terminal.log`
- internal project log: `outputs/<experiment>/<chunk>/logs/run.log`

This avoids bash job-control suspensions such as `Stopped` when a background
worker writes progress bars or tries to touch the terminal.

## Run Chunks

Example for physical GPUs 3, 4, 5, 6, and 7:

```bash
cd /data1/data2/csy/exam2

bash scripts/launch_chunked_eval.sh \
  --gpus 3,4,5,6,7 \
  --experiment-name mvbench_ours_keep040_fps4_max110 \
  --output-dir outputs \
  --save-heatmap-first-chunk true \
  --heatmap-save-count 20 \
  -- \
  --dataset mvbench \
  --method ours \
  --model-path models/LLaVA-Video-7B-Qwen2-Video-Only \
  --llava-root third_party/LLaVA-NeXT \
  --num-frames 110 \
  --frame-sampling fps \
  --video-fps 4 \
  --force-sample false \
  --keep-ratio 0.4 \
  --window-size 4 \
  --missing-video-policy skip
```

`Ctrl+C` still stops all launched chunk workers.

## Watch Logs

The terminal-facing log is useful for startup errors:

```bash
tail -f outputs/mvbench_ours_keep040_fps4_max110/chunk2/logs/terminal.log
```

The project log contains the same run output captured by `run_eval.py`:

```bash
tail -f outputs/mvbench_ours_keep040_fps4_max110/chunk2/logs/run.log
```

## Merge Chunks

Merge after all processes finish:

```bash
python scripts/merge_runs.py \
  --output-dir outputs/mvbench_ours_keep040_fps4_max110/merged \
  --run-dirs \
    outputs/mvbench_ours_keep040_fps4_max110/chunk0 \
    outputs/mvbench_ours_keep040_fps4_max110/chunk1 \
    outputs/mvbench_ours_keep040_fps4_max110/chunk2 \
    outputs/mvbench_ours_keep040_fps4_max110/chunk3 \
    outputs/mvbench_ours_keep040_fps4_max110/chunk4

cat outputs/mvbench_ours_keep040_fps4_max110/merged/summary.txt
```

The merged directory writes `logs/merge.log`, copies chunk logs into
`logs/chunks/`, and contains `summary.txt` for direct result inspection. It also
contains `skipped.jsonl` for missing or unreadable media records.
