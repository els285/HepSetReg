# hepsetreg

Regresses observables from ttbar dilepton events (reco jets, leptons, MET)
using a transformer encoder over the variable-length set of objects in each
event. Targets range from system-level kinematics (ttbar mass, spin
correlation observables) to per-particle four-momenta (top/antitop), and
training can use a plain regression head, a cross-attention decoder for
per-object prediction, or a conditional flow-matching head.

## What's here

```
src/hepsetreg/
  datasets/    parquet-backed Dataset/DataModule classes
  backbones/   encoders, decoder, regression heads, flow-matching velocity fields
  losses/      per-event regression losses and batch shape-matching losses
  training/    LightningModules wiring a model + loss into a training loop
  physics/     spin-correlation observables and ROOT -> parquet data prep
  runs/        standalone train/predict scripts, one per model + task
```

**Datasets** (`datasets/`) read parquet files with an `inputs` group
(global features, jets, electrons, muons) and a `targets` group, z-score
everything, and pad jets to a fixed count per batch.
- `EventDatasetBase.py` -- `EventDataset` / `EventDataModule`. The shared
  base: fixed global features, padded jets, exactly-two leptons, one target
  vector per event.
- `EventDatasetSlot.py` -- `SlotEventDataset` / `SlotEventDataModule`. Same
  inputs, but the target is per-slot: index 0 is always the top, index 1
  the antitop (a fixed physical order, not an arbitrary one).
- `EventVectorScalarDataset.py` -- splits jet/lepton features into
  4-momentum ("vector") and auxiliary ("scalar") groups, for backbones that
  need real physical momenta to compute pairwise quantities.

**Backbones** (`backbones/`):
- `backbone.py` / `backbone_cls.py` -- `TransformerBackbone` (attention
  pooling) and `TransformerBackboneCLS` (CLS-token pooling). Plain
  self-attention over global + jet + lepton tokens.
- `backbone_ParT.py` -- `TransformerBackboneParT`, which adds a
  Particle-Transformer-style pairwise attention bias computed from each
  jet/lepton's 4-momentum (angular separation, relative pT, pairwise
  invariant mass), on top of the same token setup.
- `decoder.py` -- `CrossAttentionDecoder`: N fixed learned queries
  cross-attending over the encoded tokens, for predicting N objects with a
  fixed, known order (e.g. top then antitop).
- `slot_attention.py` -- `SlotAttention` / `FixedSlotAttention`, an
  alternative to the decoder for permutation-invariant or fixed-slot
  per-object prediction.
- `head.py` -- `RegressionHead` (MLP) plus the wrapper classes that combine
  a backbone (+ decoder/slot-attention) with a head: `EventRegressor`,
  `SetEventRegressor`, `SlotEventRegressor`.
- `flow_matching.py` -- conditional flow matching: three interchangeable
  velocity-field architectures (`DNNVelocityField`, `TransformerVelocityField`,
  `DiTVelocityField`), `flow_matching_loss`, `sample_flow` (ODE sampler), and
  `FlowRegressor` (backbone + velocity field).

**Losses** (`losses/`):
- `losses.py` -- `RegressionLoss` (MSE/Huber/MAE) and `SetRegressionLoss`
  (Hungarian matching, for permutation-invariant set prediction).
- `shape_losses.py` -- distances between the *distribution* of predictions
  and targets over a batch, not just the per-event error: `emd`, `mmd`,
  `kl` (per output column), plus `mmd_joint` and `sliced_emd` (matching the
  joint distribution across columns, so correlations between targets are
  constrained too).

**Training** (`training/`):
- `lightning_module.py` -- `EventRegressionModule` (point regression) and
  `FlowMatchingModule` (flow matching, with ODE-sampled validation/predict).
  Both support an optional LR scheduler via `scheduler_fn`.
- `shape_module.py` -- `ShapeRegressionModule`, `EventRegressionModule` plus
  a shape-matching term from `shape_losses.py`.

**Physics** (`physics/`):
- `reco_spins.py` -- boosts tops/leptons into the ttbar rest frame and
  computes the helicity-basis spin-correlation observables (`cos_phi`,
  `cos_han`, and the K/N/R projections).
- `prep_data.py` -- builds the parquet files this package reads, from ROOT
  ntuples.

**Runs** (`runs/`) -- each script is self-contained: a `build_model`, a
`train`, a `predict`, and a config dict defined directly in `if __name__ ==
"__main__":`. No argparse beyond the `--train`/`--predict` switch; edit the
config dict in the file to change hyperparameters.

| script | backbone | task |
|---|---|---|
| `run_regression.py` | `TransformerBackbone` | system-level targets (ttbar mass, spin observables) |
| `run_regression_ParT.py` | `TransformerBackboneParT` | same task, with the pairwise attention bias |
| `run_regression_flow.py` | `TransformerBackbone` + flow matching | same task, sampled rather than predicted directly |
| `run_slot_model.py` | `TransformerBackbone` + `CrossAttentionDecoder` | per-object (top, antitop) four-momenta |
| `run_slot_model_aim.py` | same as `run_slot_model.py` | same, logged to Aim instead of TensorBoard |

Every script caps `torch` to 4 threads and sets TF32 matmul precision
regardless of model size -- this machine has many CPU cores, and letting
PyTorch use all of them for small per-op work is much slower, not faster.
`compile_model: True` in the config wraps the model in `torch.compile`;
`precision="bf16-mixed"` is set in the `Trainer` for GPU runs.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates `.venv` and installs everything needed to run the scripts
below (torch, pytorch-lightning, awkward, pyarrow, uproot, vector, aim,
...). If you also want to work in the notebooks or plot things
(`ipykernel`, `jupyter-client`, `matplotlib`) or run the test suite
(`pytest`), use:

```bash
uv sync --extra dev
```

`uv sync` without `--extra dev` will *remove* those packages if they're
already installed, since it makes the environment match `pyproject.toml`
exactly -- run `uv sync --extra dev` again afterwards if that happens.

## Running things with `uv run`

Each run script is a module inside the `hepsetreg` package, so it's run
with `-m`, not as a bare file path:

```bash
uv run python -m hepsetreg.runs.run_regression --train
uv run python -m hepsetreg.runs.run_regression --predict
```

Same pattern for the others:

```bash
uv run python -m hepsetreg.runs.run_regression_ParT --train
uv run python -m hepsetreg.runs.run_slot_model --train
uv run python -m hepsetreg.runs.run_slot_model_aim --train
uv run python -m hepsetreg.runs.run_regression_flow --train
```

Before `--predict` will work, open the script and fill in the checkpoint
and stats paths it printed at the end of training (Lightning names
checkpoints `epoch=X-step=Y.ckpt`).

To build the parquet files from ROOT ntuples (edit the path/filenames at
the bottom of the file first):

```bash
uv run python -m hepsetreg.physics.prep_data
```

To look at a finished or running training job:

```bash
uv run tensorboard --logdir tb_logs          # run_regression / run_regression_ParT / run_regression_flow / run_slot_model
uv run aim up --repo aim_logs                # run_slot_model_aim
```

### Interactive GPU session

<!-- fill in: how to request an interactive GPU node on this cluster -->

## Running via Slurm

`submit_run_regression` is a template Slurm submit script. It requests one
GPU, activates nothing (it calls `.venv/bin/python` directly), and runs:

```bash
.venv/bin/python -m hepsetreg.runs.run_regression --train
```

Submit it with:

```bash
sbatch submit_run_regression
```

For a different script, copy the file and change the partition/job name at
the top and the module path on the last line, e.g.:

```bash
.venv/bin/python -m hepsetreg.runs.run_slot_model --train
```

Check `--mem` before submitting a training run on the full dataset -- the
train parquet is loaded into memory in one go when the dataset is built.
