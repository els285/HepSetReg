"""Generates a synthetic dataset for smoke-testing hepsetreg end-to-end
without any ATLAS data.

Each event has a *random* number of jets (2-6, the thing DIRECTOR's fixed
``feature_groups`` column slicing couldn't handle), one lepton, and one MET
object. The regression target is a 2-vector ``[system_mass, system_pt]`` of
the (jets + lepton) four-vector sum -- loosely mimicking a ttbar-like
"regress an invariant mass from final-state objects" problem. A redundant
``truth_mass`` extra is stored alongside the target purely to demonstrate the
pluggable physics-consistency loss hook in ``examples/physics_hooks.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hepsetreg.data.preprocessing import pad_and_mask, write_padded_hdf5, write_scaler_hdf5


def _four_vector(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, mass: float = 0.0) -> np.ndarray:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.stack([px, py, pz, E], axis=-1)


def _make_events(n_events: int, rng: np.random.Generator, min_jets: int, max_jets: int):
    jets_list = []
    leptons = np.zeros((n_events, 1, 4), dtype=np.float32)
    met = np.zeros((n_events, 1, 2), dtype=np.float32)
    targets = np.zeros((n_events, 2), dtype=np.float32)
    truth_mass = np.zeros((n_events, 1), dtype=np.float32)

    for i in range(n_events):
        n_jets = int(rng.integers(min_jets, max_jets + 1))
        pt = rng.exponential(40.0, size=n_jets) + 20.0
        eta = rng.uniform(-2.5, 2.5, size=n_jets)
        phi = rng.uniform(-np.pi, np.pi, size=n_jets)
        jets_p4 = _four_vector(pt, eta, phi)
        btag = (rng.uniform(size=n_jets) < 0.3).astype(np.float32)
        jets_list.append(
            np.concatenate([np.stack([pt, eta, phi, jets_p4[:, 3]], axis=-1), btag[:, None]], axis=-1).astype(
                np.float32
            )
        )

        lep_pt = rng.exponential(30.0) + 20.0
        lep_eta = rng.uniform(-2.4, 2.4)
        lep_phi = rng.uniform(-np.pi, np.pi)
        lep_p4 = _four_vector(np.array([lep_pt]), np.array([lep_eta]), np.array([lep_phi]))[0]
        leptons[i, 0] = [lep_pt, lep_eta, lep_phi, lep_p4[3]]

        met[i, 0] = [rng.exponential(25.0), rng.uniform(-np.pi, np.pi)]

        system = jets_p4.sum(axis=0) + lep_p4
        mass2 = system[3] ** 2 - (system[0] ** 2 + system[1] ** 2 + system[2] ** 2)
        mass = float(np.sqrt(max(mass2, 1.0)))
        pt_system = float(np.hypot(system[0], system[1]))

        targets[i] = [mass, pt_system]
        truth_mass[i, 0] = mass

    return jets_list, leptons, met, targets, truth_mass


def _build_split(n_events: int, seed: int, min_jets: int, max_jets: int):
    rng = np.random.default_rng(seed)
    jets_list, leptons, met, targets, truth_mass = _make_events(n_events, rng, min_jets, max_jets)
    jets_features, jets_mask = pad_and_mask(jets_list, max_count=max_jets, feature_dim=5)
    groups = {
        "jets": (jets_features, jets_mask),
        "leptons": (leptons, np.ones((n_events, 1), dtype=bool)),
        "met": (met, np.ones((n_events, 1), dtype=bool)),
    }
    return groups, targets, {"truth_mass": truth_mass}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="examples/toy_data")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=800)
    parser.add_argument("--min-jets", type=int, default=2)
    parser.add_argument("--max-jets", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_groups, train_targets, train_extras = _build_split(args.n_train, args.seed, args.min_jets, args.max_jets)
    val_groups, val_targets, val_extras = _build_split(args.n_val, args.seed + 1, args.min_jets, args.max_jets)
    test_groups, test_targets, test_extras = _build_split(args.n_test, args.seed + 2, args.min_jets, args.max_jets)

    mean = train_targets.mean(axis=0)
    scale = train_targets.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)

    def scale_targets(t):
        return ((t - mean) / scale).astype(np.float32)

    write_padded_hdf5(out_dir / "train.h5", train_groups, scale_targets(train_targets), extras=train_extras)
    write_padded_hdf5(out_dir / "val.h5", val_groups, scale_targets(val_targets), extras=val_extras)
    write_padded_hdf5(out_dir / "test.h5", test_groups, scale_targets(test_targets), extras=test_extras)
    write_scaler_hdf5(out_dir / "scaler.h5", mean, scale)

    print(f"Wrote toy dataset (train={args.n_train}, val={args.n_val}, test={args.n_test}) to {out_dir}")


if __name__ == "__main__":
    main()
