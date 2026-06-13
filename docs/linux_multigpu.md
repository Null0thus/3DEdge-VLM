# Linux Multi-GPU Evaluation

Run one chunk per GPU for data-parallel evaluation. Example for physical GPUs
3, 4, 5, 6, and 7. The command group installs a trap so `Ctrl+C` stops all
background chunk processes together.

For cleaner output layout, pass the same `--experiment-name` to all chunks and
use short `--run-name` values such as `chunk0`; outputs then live under one
folder like `outputs/<experiment-name>/chunk0` and `outputs/<experiment-name>/merged`.

```bash
cd /data1/data2/csy/exam2

(
  trap 'jobs -pr | xargs -r kill; wait; exit 130' INT TERM

  CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.4 --window-size 4 --missing-video-policy skip --save-heatmap true --heatmap-save-count 20 --num-chunks 5 --chunk-idx 0 --chunk-strategy round_robin --output-dir outputs --experiment-name mvbench_ours_keep040_fps4_max110 --run-name chunk0 &
  CUDA_VISIBLE_DEVICES=4 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.4 --window-size 4 --missing-video-policy skip --save-heatmap false --num-chunks 5 --chunk-idx 1 --chunk-strategy round_robin --output-dir outputs --experiment-name mvbench_ours_keep040_fps4_max110 --run-name chunk1 &
  CUDA_VISIBLE_DEVICES=5 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.4 --window-size 4 --missing-video-policy skip --save-heatmap false --num-chunks 5 --chunk-idx 2 --chunk-strategy round_robin --output-dir outputs --experiment-name mvbench_ours_keep040_fps4_max110 --run-name chunk2 &
  CUDA_VISIBLE_DEVICES=6 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.4 --window-size 4 --missing-video-policy skip --save-heatmap false --num-chunks 5 --chunk-idx 3 --chunk-strategy round_robin --output-dir outputs --experiment-name mvbench_ours_keep040_fps4_max110 --run-name chunk3 &
  CUDA_VISIBLE_DEVICES=7 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.4 --window-size 4 --missing-video-policy skip --save-heatmap false --num-chunks 5 --chunk-idx 4 --chunk-strategy round_robin --output-dir outputs --experiment-name mvbench_ours_keep040_fps4_max110 --run-name chunk4 &
  wait
)
```

Watch a chunk log while it is running:

```bash
tail -f outputs/mvbench_ours_keep040_fps4_max110/chunk2/logs/run.log
```

Merge the five chunk outputs after all processes finish:

```bash
python scripts/merge_runs.py \
  --output-dir outputs/mvbench_ours_keep040_fps4_max110/merged \
  --run-dirs \
    outputs/mvbench_ours_keep040_fps4_max110/chunk0 \
    outputs/mvbench_ours_keep040_fps4_max110/chunk1 \
    outputs/mvbench_ours_keep040_fps4_max110/chunk2 \
    outputs/mvbench_ours_keep040_fps4_max110/chunk3 \
    outputs/mvbench_ours_keep040_fps4_max110/chunk4
```

Each chunk writes `logs/run.log` inside its own output folder. The merged
directory writes `logs/merge.log`, copies chunk logs into `logs/chunks/`, and
contains `summary.txt` for direct result inspection. It also contains
`skipped.jsonl` for missing or unreadable media records. Use
`--missing-video-policy error` when the dataset is fully prepared and you want
the first bad media path to stop the run.
