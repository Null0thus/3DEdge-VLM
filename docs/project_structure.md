# Project Structure

This project keeps third-party code, model weights, datasets, and our pruning
code separated.

```text
datasets/                 Raw datasets. Read only during experiments.
models/                   Local model weights. Read only during experiments.
third_party/LLaVA-NeXT/    Upstream LLaVA-NeXT source with a minimal hook.
configs/                  Default experiment and dataset path configs.
st_edge_pruning/          Our method, dataset loaders, metrics, and IO.
scripts/                  Evaluation and single-sample inspection entrypoints.
outputs/                  Per-run predictions, token stats, timings, heatmaps.
docs/                     Notes about structure and commands.
```

The pruning hook only receives pooled ordinary video tokens after `P` and
spatial `Pool`. Text tokens and prompt tokens remain outside the pruning code.

Each run writes `predictions.jsonl`, `token_stats.jsonl`, `timings.jsonl`,
`skipped.jsonl`, `summary.json`, `args.json`, and optional heatmaps.
