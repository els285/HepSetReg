"""
Run a trained EventRegressor checkpoint over a parquet file and produce
physical-unit predictions for TARGET_FEATURES.

Needs the `stats.pt` companion file `train.py` saves alongside each run's
checkpoints (same log_dir/version_N) -- refitting z-score stats on new data,
or padding jets to a different max_jets, would silently disagree with what
the model was actually trained on.
"""

from __future__ import annotations

import awkward as ak
import torch
from torch.utils.data import DataLoader

from backbone import TransformerBackbone
from EventDataset5 import GLOBAL_FEATURES, JET_FEATURES, LEPTON_FEATURES, TARGET_FEATURES
from EventDataset6 import EventDataset, _identity_collate
from head import EventRegressor, RegressionHead
from lightning_module import EventRegressionModule


def build_model(d_model=128, nhead=8, num_layers=6):
    """Must match the architecture the checkpoint was trained with -- see
    train.py's `main()` for the current defaults."""
    backbone = TransformerBackbone(
        n_global_features=len(GLOBAL_FEATURES),
        n_jet_features=len(JET_FEATURES),
        n_lepton_features=len(LEPTON_FEATURES),
        d_model=d_model, nhead=nhead, num_layers=num_layers,
    )
    head = RegressionHead(d_model=d_model, output_dim=len(TARGET_FEATURES))
    return EventRegressor(backbone, head)


def load_checkpoint(ckpt_path, stats_path, d_model=128, nhead=8, num_layers=6, device="cpu"):
    model = build_model(d_model, nhead, num_layers)
    module = EventRegressionModule(model)

    # if the checkpoint was saved from a torch.compile(model)-wrapped run
    # (train.py's compile_model=True), every parameter key is prefixed with
    # "_orig_mod." (that's where torch.compile keeps the real, uncompiled
    # module) -- strip it so it matches this plain, uncompiled model's keys.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in ckpt["state_dict"].items()}
    module.load_state_dict(state_dict)
    module.eval()

    saved = torch.load(stats_path, map_location="cpu")
    return module, saved["stats"], saved["max_jets"]


@torch.no_grad()
def predict(module, stats, max_jets, data_path, batch_size=4096, device="cpu"):
    """Returns (predictions, targets), both (N, len(TARGET_FEATURES)) in
    PHYSICAL units (unscaled) -- the model itself predicts in z-scored
    space, matching how it was trained."""
    ds = EventDataset(data_path, stats=stats, max_jets=max_jets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_identity_collate)

    module = module.to(device)
    t_mean, t_std = stats["target"]
    t_mean = torch.as_tensor(t_mean, device=device)
    t_std = torch.as_tensor(t_std, device=device)

    all_preds, all_targets = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred_scaled = module.model(batch)
        all_preds.append((pred_scaled * (t_std + 1e-8) + t_mean).cpu())
        all_targets.append((batch["target"] * (t_std + 1e-8) + t_mean).cpu())

    return torch.cat(all_preds), torch.cat(all_targets)


def save_predictions(preds, targets, output_path):
    """Writes per-event predictions and targets to a parquet file via
    awkward, as two record groups ("targets"/"predictions"), each with one
    field per TARGET_FEATURES entry -- e.g. `arr.targets.ttbar_mass`,
    `arr.predictions.ttbar_mass`."""
    # .contiguous(): a column slice of a 2D tensor is a strided, non-
    # contiguous view, which pyarrow (used internally by to_parquet) rejects
    # with "ndarray is not contiguous"
    target_fields = {name: ak.Array(targets[:, i].contiguous().numpy()) for i, name in enumerate(TARGET_FEATURES)}
    pred_fields = {name: ak.Array(preds[:, i].contiguous().numpy()) for i, name in enumerate(TARGET_FEATURES)}

    arr = ak.zip({"targets": ak.zip(target_fields), "predictions": ak.zip(pred_fields)}, depth_limit=1)
    ak.to_parquet(arr, output_path)


def main(
    checkpoint, stats, data="ttbar-2L_test.root_180926.parquet", batch_size=4096,
    device="cuda" if torch.cuda.is_available() else "cpu",
    d_model=128, nhead=8, num_layers=6, output=None,
):
    module, stats, max_jets = load_checkpoint(checkpoint, stats, d_model, nhead, num_layers, device)
    preds, targets = predict(module, stats, max_jets, data, batch_size, device)
    
    
    # print(ak.Array(targets.detach().cpu().numpy()))
    # input() 
    
    if output:
        save_predictions(preds, targets, output)
        print(f"saved predictions to {output}")

    return preds, targets


if __name__ == "__main__":
    # fill in the actual run's version_N and checkpoint filename (Lightning
    # names it "epoch=X-step=Y.ckpt" by default) from your training output
    main(
        checkpoint="tb_logs/event_regression/version_36/checkpoints/epoch=19-step=22920.ckpt",
        stats="tb_logs/event_regression/version_36/stats.pt",
        device="cuda",
        data="ttbar-2L_test.root_180926.parquet",
        output="ttbar-2L-inference-220926.parquet"
    )
