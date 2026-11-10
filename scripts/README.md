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

`--loss-guidance <json>` (optional, `coord_mse` only) restricts the diversity objective to chosen
atom pairs, align-free. The JSON holds a `guidances` block in the boltz2 / Protenix guidance format
(a whole input JSON with that key works too):

```json
"guidances": [{"particle": {"atom_sets": [[["A", "74:87", "CA"], ["A", "255:268", "CA"]]]}}]
```

Each atom set is `[chain, residue, atom]` entries (`"start:end"` is half-open); every atom pair
within a set is used. The loss is then the mean over ConforNet pairs of the mean squared
difference of those pair distances (squared distance-RMSD), with no superposition.

`atom_set_pairs: [[set_a, set_b], ...]` (ConforNets extension, same entry) adds only the cross
pairs a x b. Example, TM6 intracellular end (D6IC residues) against a TM1/2/4 anchor:

```json
"guidances": [{"particle": {"atom_set_pairs": [[
    [["A", "245:253", "CA"]],
    [["A", "3:25", "CA"], ["A", "43:65", "CA"], ["A", "122:140", "CA"]]
]]}}]
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
