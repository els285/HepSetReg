# hepsetreg

A small PyTorch Lightning package for **regressing observables (e.g. the
ttbar invariant mass) from a variable-length set of reconstructed physics
objects** (jets, leptons, MET, ...) using a transformer encoder.

It exists to fix the main limitation of the `DIRECTOR` prototype
(https://github.com/AlexeiM2004/DIRECTOR) while reusing its good ideas
(MHA/MLA-style transformer regression, flow-matching training, KL-divergence
distribution matching, a physics-consistency loss term) and the masked-token
transformer pattern from `ReconstructionAndSB`
(https://github.com/diegobaronm/ReconstructionAndSB):

| | DIRECTOR | hepsetreg |
|---|---|---|
| Jets per event | Fixed `feature_groups` column slices in a flat `X` vector; changing the jet count means re-deriving column indices and retraining the projection layers. | Any number of jets (or other repeated objects) per event, handled by padding + an attention mask. One linear projection is shared across every instance of an object type. |
| Loss functions | Huber + KL + a ttbar-specific hard-coded mass formula, all wired together in one training script. | Modular named terms (`regression`, `distribution_kl`, `flow_matching`, `consistency`) combined by `CompositeLoss`, each with its own weight and optional ramp -- include any subset. |
| Physics-consistency loss | A specific formula reconstructing top/antitop/ttbar mass from 8 fixed output columns. | A user-supplied callable (`PhysicsConsistencyLoss(fn)`); works for any process/target layout. |
| Training loop | Hand-written `for epoch in range(...)` loop, duplicated between `transformer_train.py` and `flowmatch_train.py`. | `pytorch_lightning.LightningModule` + `Trainer`, standard checkpointing/early-stopping/logging. |

## Install

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10. Core deps: `torch`, `lightning`, `numpy`, `h5py`,
`pyyaml`, `omegaconf`. Dev deps (`pytest`, `matplotlib`) are only needed to
run the test suite / plot things yourself.

## Quick start (toy dataset, no ATLAS data required)

```bash
python examples/make_toy_dataset.py           # writes examples/toy_data/{train,val,test}.h5 + scaler.h5
PYTHONPATH=examples hepsetreg-train --config configs/example_regression.yaml
PYTHONPATH=examples hepsetreg-train --config configs/example_flow_matching.yaml
```

(`PYTHONPATH=examples` is only needed so the config's
`loss.terms.consistency.fn: "physics_hooks:toy_mass_consistency"` dotted
path can be imported -- see "Physics-consistency hook" below.)

## Core ideas

### 1. Objects are grouped by type, not by fixed column position

Instead of one flat feature vector, an event is a `dict` of object groups:

```python
objects = {
    "jets":    torch.Tensor(B, max_jets, jet_feature_dim),     # zero-padded
    "leptons": torch.Tensor(B, max_leptons, lepton_feature_dim),
    "met":     torch.Tensor(B, 1, met_feature_dim),
}
mask = {
    "jets":    torch.BoolTensor(B, max_jets),     # True = real object
    "leptons": torch.BoolTensor(B, max_leptons),
    "met":     torch.BoolTensor(B, 1),
}
```

`max_jets` etc. are just how wide you chose to pad -- the model places no
constraint on it, and events with fewer real jets than `max_jets` are simply
masked. `ObjectSetEncoder` applies one shared `nn.Linear` per group (a
**permutation-equivariant** projection: jet order doesn't matter) plus a
type-embedding, concatenates every group into one token sequence, and runs a
standard `nn.TransformerEncoder` with `src_key_padding_mask` built from the
masks. Pooling (`cls` / `mean` / `attention`) collapses the token sequence
into one event embedding for the regression head.

`tests/test_backbone.py` checks this directly: widening the padding (even
with garbage values in the new slots) or reordering jets within an event
leaves the pooled output numerically unchanged.

### 2. Loss terms are modular

```python
from hepsetreg.losses import CompositeLoss, LossTermConfig, RegressionLoss, HistogramKLDivergenceLoss

loss = CompositeLoss(
    terms={"regression": RegressionLoss(kind="huber"), "distribution_kl": HistogramKLDivergenceLoss(bins=100)},
    configs={
        "regression": LossTermConfig(weight=1.0),
        "distribution_kl": LossTermConfig(weight=0.2, ramp_epochs=15),  # linear warm-up, like DIRECTOR's kl_ramp_epochs
    },
)
```

`hepsetreg.losses.distribution` has four interchangeable options for the
`distribution_kl` slot (all share the `forward(pred, target) -> Tensor`
contract, so swapping one for another needs no other code changes):
`HistogramKLDivergenceLoss` (binned, per-dimension marginals -- DIRECTOR's
original approach, generalized), `KNNKLDivergenceLoss` (unbinned KL
estimator via k-nearest-neighbour distances, sees the full joint
distribution rather than marginals), `MMDLoss` (unbinned, kernel-based
Maximum Mean Discrepancy, bounded and stable for small batches), and
`SlicedWassersteinLoss` (unbinned, transport-based via random 1-D
projections, cheapest of the three unbinned options).

`SupervisedRegressor` and `FlowMatchingRegressor` both just build an
`inputs` dict per training step (`{"regression": {...}, "distribution_kl":
{...}, ...}`) and call `loss(inputs)` -- so any subset of `regression` /
`distribution_kl` / `flow_matching` / `consistency` can be active, each
independently weighted/ramped, without touching the training loop.

### 3. The physics-consistency loss is a hook, not a formula

DIRECTOR hard-codes: read 8 output columns as top/antitop four-vectors,
reconstruct 3 invariant masses, Huber-loss them against a `M` HDF5 array.
Here that's a user callable:

```python
def ttbar_mass_consistency(y_pred, batch):
    ...  # your formula, your output layout, your truth arrays
    return {"top": ..., "antitop": ..., "system": ...}   # or a single scalar Tensor

loss.terms["consistency"] = PhysicsConsistencyLoss(ttbar_mass_consistency)
```

`examples/physics_hooks.py` includes both a toy example and a literal
reimplementation of DIRECTOR's ttbar formula as a hook, for reference.

### 3b. Optional backbone: PairformerBackbone (AlphaFold3-style)

`hepsetreg.models.backbone_pairformer.PairformerBackbone` is a drop-in
alternative to `ObjectSetEncoder` (same `forward(objects, mask)` contract)
based on AlphaFold3's Pairformer: alongside the per-object single
representation, it maintains an N_obj x N_obj **pair representation** that
gets refined every block via triangle multiplicative updates (and,
optionally, triangle self-attention), and that refined pair representation
supplies an additive **bias to the single-representation attention logits**
-- so the token-to-token attention bias is itself updated layer by layer,
not fixed. It can optionally be seeded with explicit physical pairwise
features (`pairwise_features`, shape `(B, N_obj, N_obj, d)` -- e.g. ΔR/k_T
per jet pair, in the spirit of the Particle Transformer's interaction
features). It's heavier than `ObjectSetEncoder` and not wired into the YAML
config/factory yet -- swap it in directly if you want to try it:

```python
from hepsetreg.models import PairformerBackbone, RegressionHead
from hepsetreg.lightning_modules import SupervisedRegressor

encoder = PairformerBackbone(groups, d_model=128, d_pair=64, nhead_single=8, num_blocks=4)
head = RegressionHead(d_model=128, output_dim=2)
model = SupervisedRegressor(encoder, head, loss, train_cfg=train_cfg)
```

Triangle self-attention (`use_triangle_attention=True`) is off by default:
it's the most expensive and intricate part of AF3's block and, for the
small object counts typical here (a handful of jets), the triangle
multiplicative updates alone already mix information across the whole
event per block.

### 3c. Optional backbone: CovariantParticleTransformer (boost-covariant)

`hepsetreg.models.backbone_covariant.CovariantParticleTransformer` is a
lightweight backbone inspired by the [Covariant Particle
Transformer](https://github.com/shikaiqiu/Covariant-Particle-Transformer)
(Qiu et al., [arXiv:2203.05687](https://arxiv.org/abs/2203.05687)), built to
be exactly invariant to the residual symmetry hadron-collider final states
actually have: longitudinal boosts along the beam (different events are
seen with different, unknown net boosts because the two colliding partons
carry different, unknown momentum fractions) and azimuthal rotations about
the beam (an arbitrary detector convention). It works in `(pT, y, phi, m)`
instead of `(px, py, pz, E)`: `pT`/`m` are already invariant under that
symmetry, and pairwise attention is conditioned only on the *invariant*
relative descriptor `(y_j - y_i, cos(phi_j - phi_i), sin(phi_j - phi_i))` --
so the pooled event embedding this backbone produces for the regression
head is provably, and numerically (see
`tests/test_backbone_covariant.py::test_boost_and_rotation_invariance`),
unchanged if you boost/rotate every object in an event by the same amount
before running it through the network. That's the useful property for a
target like an invariant mass, which by definition shouldn't depend on
which (equally valid) boosted frame you happened to measure the event in.

Unlike the other two backbones it needs explicit four-vectors (it computes
pT/y/phi/m itself), not opaque feature vectors:

```python
from hepsetreg.models import CovariantParticleTransformer, KinematicGroupSpec, RegressionHead
from hepsetreg.lightning_modules import SupervisedRegressor

groups = [KinematicGroupSpec("jets", extra_features=1), KinematicGroupSpec("leptons")]  # +1 = b-tag score
encoder = CovariantParticleTransformer(groups, d_model=64, nhead=4, num_blocks=4, pooling="attention")
head = RegressionHead(d_model=64, output_dim=2)
model = SupervisedRegressor(encoder, head, loss, train_cfg=train_cfg)

# forward: four_vectors[name] is (B, N, 4) in (px, py, pz, E) order, mask[name] is (B, N) bool
pooled = encoder(four_vectors, mask, extra_features={"jets": btag_scores})
```

`extra_features` is only for values that are *already* invariant under
beam boosts/rotations (a b-tag score, PID, isolation) -- never put raw
px/py/pz there, that would reintroduce the frame-dependence this backbone
exists to remove. Only `pooling="mean"`/`"attention"` are supported (no
`"cls"`: a CLS token has no physical four-vector, so it doesn't fit the
relative-geometry attention mechanism). Scope note: this implements CPT's
covariant *encoder* mechanism (self-attention conditioned on invariant
relative geometry, covariant rapidity/phi updates), not their
sequence-to-sequence decoder for reconstructing unseen particles -- this
package pools to a single event embedding for regression, which doesn't
need that decoder. It also only covers the beam-axis subgroup (longitudinal
boosts + azimuthal rotations), the same "partial" covariance scope the
paper itself targets, not the full Lorentz group.

### 4. Two training modes, one data pipeline

- `SupervisedRegressor`: encoder -> pooled embedding -> MLP head -> point
  prediction, trained with `regression` (+ optional `distribution_kl` /
  `consistency`).
- `FlowMatchingRegressor`: generalizes `flowmatch_train.py` -- a
  `ConditionalVelocityField` attends over the (masked, variable-length)
  context objects plus a time token and an `x_t` token, trained with
  conditional flow matching (+ optional `distribution_kl` / `consistency`,
  computed from ODE-sampled predictions during training exactly as DIRECTOR
  does, but cheaper `sampling.n_steps_training`/`n_samples_training` for
  training vs. `sampling.n_steps`/`n_samples` for inference).

Both read the same `PaddedObjectDataset` / `RegressionDataModule` batches.

## Data format

`PaddedObjectDataset` expects an HDF5 file:

```
/{group}/features   float32 (n_events, n_group_max, n_group_features)
/{group}/mask       bool    (n_events, n_group_max)     True = real object
/target             float32 (n_events, n_targets)
/extras/{name}      any     (n_events, ...)              optional, e.g. truth arrays for a consistency hook
```

and a separate scaler file (optional, only needed if you standardise
targets): `Y_mean` / `Y_scale`, same convention DIRECTOR uses.

`hepsetreg.data.preprocessing` has two helpers to build this from
jagged/variable-length per-event arrays:

```python
from hepsetreg.data.preprocessing import pad_and_mask, write_padded_hdf5, write_scaler_hdf5

jets_features, jets_mask = pad_and_mask(list_of_per_event_jet_arrays, max_count=8)
write_padded_hdf5("train.h5", {"jets": (jets_features, jets_mask), ...}, target_array, extras={"truth_mass": ...})
write_scaler_hdf5("scaler.h5", mean, scale)
```

See `examples/make_toy_dataset.py` for a complete worked example (including
computing four-vectors and an invariant-mass-like target from scratch).

## Config-driven training

`configs/example_regression.yaml` and `configs/example_flow_matching.yaml`
are annotated end-to-end examples. The schema (see `hepsetreg/factory.py`
for exactly how each field is consumed):

```yaml
model:
  mode: supervised            # or flow_matching
  d_model: 128
  nhead: 8
  num_layers: 6
  pooling: cls                # cls | mean | attention  (supervised mode only)
  output_dim: 2
  groups:
    - {name: jets, in_features: 5}
    - {name: leptons, in_features: 4}
    - {name: met, in_features: 2}
  head: {n_layers: 3, start_neurons: 128, dropout: 0.05}   # supervised mode only

loss:
  terms:
    regression: {kind: huber, weight: 1.0}
    distribution_kl: {bins: 100, sigma: 0.4, weight: 0.2, ramp_epochs: 15}
    consistency: {fn: "my_project.physics:my_consistency_fn", weight: 0.05}

data:
  group_names: [jets, leptons, met]
  train_file: ..., val_file: ..., test_file: ...
  scaler_file: ...
  batch_size: 256

train:
  optimizer: adamw
  learning_rate: 5.0e-4
  max_epochs: 100
  early_stopping_patience: 10

sampling:   # flow_matching mode only
  n_steps: 50
  n_samples: 8
  n_steps_training: 8
  n_samples_training: 4
```

Train with `hepsetreg-train --config path/to/config.yaml [dotlist overrides...]`,
e.g. `hepsetreg-train --config configs/example_regression.yaml train.max_epochs=5`.

## Package layout

```
src/hepsetreg/
  models/
    tokenizer.py        # ObjectGroupSpec, ObjectTokenizer -- the masked-token-set core
    backbone.py          # ObjectSetEncoder (supervised backbone)
    backbone_pairformer.py # PairformerBackbone (AlphaFold3-style, optional alt. backbone)
    backbone_covariant.py   # CovariantParticleTransformer (boost-covariant, optional alt. backbone)
    pooling.py            # cls / mean / attention pooling
    heads.py                # RegressionHead MLP
    flow_matching.py         # ConditionalVelocityField + sample_flow (ODE sampler)
  losses/
    regression.py         # Huber / MSE / MAE
    distribution.py         # HistogramKLDivergenceLoss / KNNKLDivergenceLoss / MMDLoss / SlicedWassersteinLoss
    flow_matching.py          # ConditionalFlowMatchingLoss
    consistency.py              # PhysicsConsistencyLoss (pluggable hook)
    composite.py                  # CompositeLoss + LossTermConfig (weights/ramps)
  lightning_modules/
    supervised.py           # SupervisedRegressor
    flow_matching.py          # FlowMatchingRegressor
  data/
    dataset.py               # PaddedObjectDataset (HDF5)
    datamodule.py              # RegressionDataModule
    scaling.py                   # TargetScaler
    preprocessing.py               # pad_and_mask / write_padded_hdf5 / write_scaler_hdf5
  config.py, factory.py, cli.py   # YAML -> objects -> Trainer
configs/          example_regression.yaml, example_flow_matching.yaml
examples/         make_toy_dataset.py, physics_hooks.py
tests/            pytest suite (see below)
```

## Tests

```bash
pytest
```

- `test_backbone.py` -- padding/permutation invariance (the core "variable
  number of jets" property), input-shape validation.
- `test_backbone_pairformer.py` -- the same padding/permutation invariance
  checks for `PairformerBackbone`, plus the triangle-attention and
  explicit-pairwise-features paths.
- `test_backbone_covariant.py` -- padding/permutation invariance for
  `CovariantParticleTransformer`, plus (the property that actually matters
  here) numerically checking that a random per-event longitudinal
  boost + azimuthal rotation applied to every object leaves the pooled
  output unchanged.
- `test_flow_matching.py` -- velocity-field output shapes, ODE sampler shape.
- `test_losses.py` -- each loss term in isolation (including the three new
  unbinned distribution losses), `CompositeLoss` weight ramping and
  term-skipping.
- `test_datamodule.py` -- HDF5 round-trip, batch shapes.
- `test_end_to_end.py` -- a `Trainer(fast_dev_run=...)` smoke run through the
  full config -> factory -> Trainer path, for both `supervised` and
  `flow_matching` modes.

> **Note on how this package was verified:** the sandbox this was built in
> had its outbound network access blocked for the whole session (`pip
> install` to PyPI returned `403` for every package, not just `torch`), so
> `torch`/`lightning`/`h5py` couldn't be installed here and the test suite
> above hasn't actually been executed yet. Every file was compiled
> (`py_compile`) and carefully re-reviewed by hand (tensor shapes, mask
> conventions, `CompositeLoss` input/signature matching across all loss
> terms). The framework-independent parts were executed and checked against
> explicit assertions where possible: `pad_and_mask`'s padding/truncation
> logic (plain NumPy), and for `CovariantParticleTransformer` specifically,
> the (pT, y, phi, m) <-> (px, py, pz, E) round-trip, the boost/rotation
> transform, the invariance of the pairwise relative descriptor, and the
> masked-attention einsum pattern (all re-derived and checked numerically in
> NumPy, since the covariance property is the entire point of that backbone
> and was worth double-checking harder than "it compiles"). Please run
> `pip install -e ".[dev]" && pytest` plus the toy-dataset quick start above
> once you have it -- flag anything that breaks and it'll get fixed.

## Extending

- **New object type**: add another entry to `model.groups` in the config
  (name + raw feature width) and make sure your HDF5 file has a matching
  `{name}/features` / `{name}/mask` pair. No code changes needed.
- **New loss term**: write an `nn.Module` whose `forward` returns a scalar
  `Tensor` (or `(scalar, {name: Tensor})` for multi-component logging), wire
  it into `hepsetreg.factory.build_loss`, and start referencing it by name
  from `loss.terms` in your config.
- **New physics-consistency check**: write a plain function
  `fn(y_pred, batch) -> Tensor | dict[str, Tensor]` and point
  `loss.terms.consistency.fn` at it (`module:function` or
  `module.function`).
