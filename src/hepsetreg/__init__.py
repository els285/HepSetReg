"""hepsetreg: a small PyTorch Lightning package for regressing observables
(e.g. the ttbar invariant mass) from a *variable-length set* of reconstructed
physics objects using a transformer encoder.

Design goals (in reaction to the hard-coded ``DIRECTOR`` prototype):

* No fixed per-slot jet columns. Any number of jets (or other repeated
  objects) per event is handled natively via padding + an attention mask,
  with a single permutation-equivariant projection shared across all
  instances of an object type.
* Loss functions (Huber/MSE regression, distribution-matching KL-divergence,
  conditional flow-matching, physics-informed consistency terms) are modular
  building blocks combined by :class:`hepsetreg.losses.CompositeLoss`, rather
  than being hard-coded together in one training script.
* The physics-consistency loss (e.g. "predicted top mass should match the
  truth top mass") is a user-supplied callable, not a formula baked in for
  ttbar specifically -- so the same package works for any process/target set.
"""

from hepsetreg.version import __version__

__all__ = ["__version__"]
