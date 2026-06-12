# Linux Multi-GPU Evaluation

Run one chunk per GPU for data-parallel evaluation. Example for physical GPUs
3, 4, 5, 6, and 7:

```bash
cd /data1/data2/csy/exam2
mkdir -p logs

CUDA_VISIBLE_DEVICES=3 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.5 --window-size 4 --save-heatmap true --heatmap-save-count 20 --num-chunks 5 --chunk-idx 0 --chunk-strategy round_robin --output-dir outputs --run-name mvbench_ours_keep050_fps4_max110_chunk0 > logs/mvbench_ours_chunk0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.5 --window-size 4 --save-heatmap false --num-chunks 5 --chunk-idx 1 --chunk-strategy round_robin --output-dir outputs --run-name mvbench_ours_keep050_fps4_max110_chunk1 > logs/mvbench_ours_chunk1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.5 --window-size 4 --save-heatmap false --num-chunks 5 --chunk-idx 2 --chunk-strategy round_robin --output-dir outputs --run-name mvbench_ours_keep050_fps4_max110_chunk2 > logs/mvbench_ours_chunk2.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.5 --window-size 4 --save-heatmap false --num-chunks 5 --chunk-idx 3 --chunk-strategy round_robin --output-dir outputs --run-name mvbench_ours_keep050_fps4_max110_chunk3 > logs/mvbench_ours_chunk3.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python scripts/run_eval.py --dataset mvbench --method ours --model-path models/LLaVA-Video-7B-Qwen2-Video-Only --llava-root third_party/LLaVA-NeXT --num-frames 110 --frame-sampling fps --video-fps 4 --force-sample false --keep-ratio 0.5 --window-size 4 --save-heatmap false --num-chunks 5 --chunk-idx 4 --chunk-strategy round_robin --output-dir outputs --run-name mvbench_ours_keep050_fps4_max110_chunk4 > logs/mvbench_ours_chunk4.log 2>&1 &
wait
```

Watch a chunk log while it is running:

```bash
tail -f logs/mvbench_ours_chunk2.log
```

Merge the five chunk outputs after all processes finish:

```bash
python scripts/merge_runs.py \
  --output-dir outputs/mvbench_ours_keep050_fps4_max110_merged \
  --run-dirs \
    outputs/mvbench_ours_keep050_fps4_max110_chunk0 \
    outputs/mvbench_ours_keep050_fps4_max110_chunk1 \
    outputs/mvbench_ours_keep050_fps4_max110_chunk2 \
    outputs/mvbench_ours_keep050_fps4_max110_chunk3 \
    outputs/mvbench_ours_keep050_fps4_max110_chunk4
```
