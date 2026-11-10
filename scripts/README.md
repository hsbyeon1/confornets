# Scripts

Benchmark entry points for inference, evaluation, and result summarization. Assumes the `assets/` scaffolding has been preprocessed.

## Shared CLI helpers

Argparse helpers live in `confornet/utils/cli.py` and are composed by every script:

- `add_model_args`: `--checkpoint` (required), `--config-yaml`, `--output-dir` (required), `--num-recycles`.
- `add_benchmark_args`: `--benchmark` (required), `--assets-dir` (default `./assets`), `--test-case` (comma-separated; default all), `--num-nodes`, `--node-idx` (inter-node sharding; see top-level README).
- `add_confidence_args`: `--compute-confidence` (write per-sample plddt/gpde/ptm CSV), `--save-full-confidence` (also dump raw tensors; implies `--compute-confidence`).


## run_diversity.py

Diversity training and sampling. Trains multiple ConforNets jointly to produce diverse conformational samples.

```bash
torchrun --nproc_per_node=4 -m scripts.run_diversity \
    --benchmark domainmotion \
    --test-case P69441 \
    --checkpoint /path/to/of3-p2-155k.pt \
    --k-confornets 4 --num-runs 12 \
    --output-dir ./output/diversity
```

`--objective` picks the loss:
- `coord_mse` (default) — pairwise L2 between per-ConforNet 1-step diffusion rollouts.
- `dist_cdf_mse` — pairwise L2 between per-ConforNet distogram CDFs; no rollout needed.

Residue-specific `coord_mse` (optional; default is the whole structure):

- `--loss-frame-res` — residue ranges whose CA atoms define the per-pair Kabsch superposition
  (the reference frame). These atoms set the frame only; the loss is not taken over them.
- `--loss-res` — residue ranges whose atoms the pairwise MSE is computed over (what diversity is
  maximised on). `--loss-atoms ca|heavy` picks CA only (default) or all heavy atoms.
- `--loss-chain` — chain the ranges refer to (default `A`). Ranges use the query's `res_id`
  numbering, i.e. the residue numbers written to the output CIFs; `a-b` is inclusive.

Either flag alone falls back to all atoms for the other. Example: diversify the intracellular end
of TM6 relative to a TM1/2/4 frame:

```bash
python -m scripts.run_diversity ... \
    --loss-frame-res 3-24,43-64,122-139 --loss-res 245-252
```


## run_mse_training.py

MSE training toward reference structures. Trains one ConforNet per (test case, reference).

```bash
torchrun --nproc_per_node=4 -m scripts.run_mse_training \
    --benchmark domainmotion \
    --test-case P69441 \
    --checkpoint /path/to/of3-p2-155k.pt \
    --num-runs 12 \
    --output-dir ./output/mse
```

## run_transfer.py

Inference with previously trained ConforNet(s). Two modes: single ConforNet or auto-discovery from an MSE training output directory.

### Single ConforNet mode

```bash
python -m scripts.run_transfer \
    --confornet-path ./output/mse/domainmotion/P69441/ref1/run_0/confornet.pt \
    --benchmark domainmotion \
    --test-case P69441 \
    --checkpoint /path/to/of3-p2-155k.pt \
    --num-samples 5 \
    --output-dir ./output/transfer
```

### Auto-discovery mode

Scans an `--mse-dir` for trained ConforNets and applies them to target test cases.

```bash
torchrun --nproc_per_node=4 -m scripts.run_transfer \
    --mse-dir ./output/mse \
    --benchmark domainmotion \
    --source P69441 \
    --test-case P69441,P12345 \
    --checkpoint /path/to/of3-p2-155k.pt \
    --output-dir ./output/transfer
```

Extras: `--confornet-path`, `--mse-dir`, `--source` (comma-separated sources; default all), `--num-samples`. Targets come from the shared `--test-case` flag.

## run_baseline.py

Baseline OF3p diffusion sampling. Reference distribution.

```bash
torchrun --nproc_per_node=4 -m scripts.run_baseline \
    --benchmark domainmotion \
    --checkpoint /path/to/of3-p2-155k.pt \
    --num-seeds 20 --num-samples 5 \
    --output-dir ./output/baseline
```

## evaluate.py

Evaluates sampled structures against reference conformations (USAlign RMSD) and pickles an `EvalResults` object.

```bash
python -m scripts.evaluate \
    --samples-dir ./output/diversity \
    --benchmark domainmotion \
    --mode diversity \
    --assets-dir ./assets \
    --output ./output/diversity/results.pkl
```

## summarize.py

Takes one or more `EvalResults` pickles and emits summary plots / tables.

```bash
python -m scripts.summarize \
    --input ./output/diversity/results.pkl \
    --output ./output/diversity/summary
```
