# Simplified conditional optimal transport for regression correction

This implementation is a **simplified conditional OT (COT) regulariser**.
It compares predicted and true distributions within fixed predicted-mass groups.
The COT experiment refines a frozen Huber + 0.1 EMD regression model for four
dileptonic top-pair observables: pair mass, rapidity difference, positive-lepton
helicity cosine and dilepton opening-angle cosine. A small trainable head
receives the frozen event representation and four baseline predictions. Its
bounded correction is added in the original training-standardised coordinates.
The head is supervised by the paired truth and a distribution penalty.

## Fixed groups

For each event, use the **frozen baseline predicted mass**. Fit the 60th and 90th
percentiles on the complete training split and persist them with the training
group counts and probabilities. The resulting populations are approximately
60%, 30% and 10%; ties can change these fractions. Equality at a boundary enters
the upper group. Boundaries and assignments do not follow the evolving corrected
prediction. Validation/test data are never used to fit the boundaries.

These groups are distinct from the physical bins used to plot migration matrices.
For each batch group, one event-index mask selects both corrected predictions
and their corresponding truth values. The truths are **not independently grouped
by true mass**. This is a coarse conditioning variable derived from detector-side
information, rather than conditioning the discrepancy on every measured input.

## What does "sliced" mean here?

Each event has four numbers: the four standardised observables. We multiply them
by a chosen set of weights and add them up, giving one score per event. We do
this for predictions and truth using the same weights, sort the two score lists
separately, and compare the lowest score with the lowest, the next with the next,
and so on. We then change the weights and repeat, averaging the resulting costs.

Each weighted sum is a one-dimensional **projection**, also called a **slice**
in sliced Wasserstein. It combines the four observables; it does not select a
subset of events or split the training dataset. Different weights examine
different combinations of the observables.

## Exact loss and settings

Write the four standardised prediction and truth as $\hat{\mathbf z}_i$ and
$\mathbf z_i$. With 32 fixed unit directions $\mathbf v_\ell$, form the scalar
projections $\mathbf v_\ell^T\hat{\mathbf z}_i$ and $\mathbf v_\ell^T\mathbf z_i$.
Within batch group $g$, sort each projected list independently and denote the
ordered entries by $a_{g\ell,(j)}$ and $b_{g\ell,(j)}$. The implemented penalty is

$$
\mathcal L_{\mathrm{COT}}=
\sum_{g:n_g\ge2}\frac{\pi_g}{32n_g}
\sum_{\ell=1}^{32}\sum_{j=1}^{n_g}
\left(a_{g\ell,(j)}-b_{g\ell,(j)}\right)^2,
$$

where $n_g$ is the batch group size and $\pi_g$ its full-training population
fraction. Empty and singleton groups contribute zero, without renormalising the
other weights. The batch objective is not asserted to be an unbiased estimate of
the population transport distance. Standardisation is fitted on training data
and keeps the mass coordinate from dominating merely because it is measured in
GeV. The reference implementation uses float32 loss arithmetic, including under
mixed-precision training, and fixed projection seed 4201.

Sorting pairs ranks instead of event identities. The stored data order is
unchanged. Gradients flow back to each contributing prediction; the target is
detached. A permutation of predictions within a group can leave this loss at zero
while producing large event errors. The paired regression term is therefore
retained. In the correction experiment the total objective is

$$
\mathcal L=\mathcal L_{\mathrm{base}}+0.1\mathcal L_{\mathrm{COT}},
$$

where $\mathcal L_{\mathrm{base}}=2\mathcal L_{\mathrm{Huber},\delta=1}+0.1\mathcal L_{\mathrm{marginal}\,W_1}$, with mean reduction over batch events and four coordinates. The factor of two preserves the recorded correction experiment.
The transport component is a finite-projection, grouped sliced-$W_2^2$ penalty
in standardised coordinates. The evaluation metric is marginal $W_1$ in physical
coordinates. Their different powers, dimensions and units serve different roles.

## How does this differ from full four-dimensional transport?

The key difference is the matching inside each group. For equally sized,
equally weighted batches, **full four-dimensional OT finds one matching** between
predicted and true events that minimises the total squared differences across
all four standardised observables. Every coordinate contributes to that same
matching. More general sample weights use a transport coupling.

**Our simplified calculation finds a separate sorted matching for each score.**
A predicted event can be matched to one truth event for one set of weights and
to a different truth event for another. Sorting these one-number lists is cheap,
but their average cost is a different objective from finding one joint matching
in four dimensions. A finite set of scores can also miss differences between
the joint distributions. Adding more scores improves coverage; the resulting
distance remains sliced Wasserstein rather than the full four-dimensional
Wasserstein distance.

Both calculations can use the same mass groups. Using richer detector
information to define the conditions would be a separate extension; it is not
a requirement for doing full four-dimensional transport. Likewise, freezing the
encoder and bounding the correction head are model choices in this experiment.
The paired Huber term keeps each prediction tied to its own event's truth while
the transport term compares distributions.

**"Full transport" and "full dataset" mean different things.** Full transport
here means comparing all four coordinates together in one transport problem.
The recorded simplified-COT run already uses all 4,692,288 training events and
260,190 validation events. The computational saving comes from the simpler
distribution comparison, not from using fewer events.

## Using the module

```python
import json
from hepsetreg.losses.conditional_ot import fit_groups, assign_groups, make_directions, conditional_ot_loss

reference = fit_groups(full_training_baseline_mass)
with open("groups.json", "w") as stream:
    json.dump(reference.to_dict(), stream, indent=2)

groups = assign_groups(batch_baseline_mass, reference)
directions = make_directions().to(prediction_z.device)
cot = conditional_ot_loss(prediction_z, target_z, groups,
                          reference.probabilities, directions)
loss = base_loss + 0.1 * cot
loss.backward()
```

Save `reference.to_dict()` and the direction tensor with the model checkpoint.
Restore group metadata using `GroupDefinition.from_dict(...)`; do not refit it
on validation or test events. The Python API validates metadata, but the caller
remains responsible for passing only training rows to `fit_groups`.

Run the synthetic example with `python examples/conditional_ot_correction.py` after installing
the package. It demonstrates a frozen baseline, a bounded two-layer correction head and all three loss terms. The input events are synthetic; replace its loader with the project data loader for real training.

## Evaluation

Evaluate all metrics on one unchanged output per event: residual quantile width,
truth-row-normalised migration matrices, marginal $W_1$, bias and physical-support
violations. Keep the original baseline and a correction-only head as separate
controls. Report low/high-mass slices and the same bin boundaries for every
method. The recorded study used validation-selected checkpoints and one seed.

## Literature

The four references, their roles in this implementation and direct paper links
are included in [conditional_ot_references.md](conditional_ot_references.md).
Ready-to-use citations are in [conditional_ot_references.bib](conditional_ot_references.bib).

- **Tabak, Trigila and Zhao (2021)**: conditional transport between distributions
  sharing covariates. [Article](https://doi.org/10.1007/s10994-021-06060-0).
- **Deshpande, Zhang and Schwing (2018)**: learning with sliced Wasserstein
  objectives. [Paper](https://arxiv.org/abs/1803.11188).
- **Rowland et al. (2019)**: projection-based Wasserstein estimation and sampling
  schemes. [Paper](https://proceedings.mlr.press/v89/rowland19a.html).
- **Peyre and Cuturi (2019)**: transport couplings, maps and numerical methods.
  [Monograph](https://arxiv.org/abs/1803.00567).

The fixed mass groups and bounded regression correction are the project-specific
design. The present projections are seeded normalised Gaussian vectors; Rowland
et al. also study alternative, orthogonally sampled directions.

## Connecting a trained HepSetReg model

`FrozenBaselineCorrection` in the example accepts an `EventRegressor` with the
existing `backbone(batch)` and `head(features)` interfaces. Load the baseline
checkpoint and its original target scaler, freeze all baseline parameters, and
keep the baseline in evaluation mode. The wrapper preserves that mode even
when the correction head is set to training mode. Use an optimiser over
`model.correction.parameters()` only.

Make one full training pass to collect immutable baseline physical masses and
fit the reference. Persist group labels in original event order or compute them
from the same frozen predictions. The standardised correction is bounded by
`0.5 * tanh(raw)`, with two 128-unit SiLU hidden layers and a zero-initialised
output layer. For the 128-feature baseline this gives 34,052 trainable parameters.
The output is exactly the baseline at initialisation.

The example uses `hepsetreg.losses.shape_losses.emd_loss` for the existing
marginal term, multiplied by 0.1. The Huber term has delta 1 and multiplier 2;
COT has multiplier 0.1. Save those settings, training mean/std, observable order,
group reference and projection tensor with the correction checkpoint. Inference
needs the baseline and correction head, but no truth or sorting operation.
For physical export, add `delta_z * target_std` to the exact baseline physical
predictions. The bound limits correction size in standardised coordinates; it
does not itself enforce cosine values inside [-1,1].

Focused checks (from the repository root, after environment setup):

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_conditional_ot.py -v
PYTHONPATH=src python examples/conditional_ot_correction.py
```
