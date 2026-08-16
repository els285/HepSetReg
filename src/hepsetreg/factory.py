"""Builds package objects (encoder/velocity-field, loss, datamodule,
LightningModule) from a loaded YAML :class:`~omegaconf.DictConfig`.

See ``configs/example_regression.yaml`` and ``configs/example_flow_matching.yaml``
for the expected schema.
"""

from __future__ import annotations

import importlib
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf

from hepsetreg.data.datamodule import RegressionDataModule
from hepsetreg.data.scaling import TargetScaler
from hepsetreg.lightning_modules.flow_matching import FlowMatchingRegressor, SamplingConfig
from hepsetreg.lightning_modules.supervised import SupervisedRegressor
from hepsetreg.losses.composite import CompositeLoss, LossTermConfig
from hepsetreg.losses.consistency import PhysicsConsistencyLoss
from hepsetreg.losses.distribution import HistogramKLDivergenceLoss
from hepsetreg.losses.flow_matching import ConditionalFlowMatchingLoss
from hepsetreg.losses.regression import RegressionLoss
from hepsetreg.models.backbone import ObjectSetEncoder
from hepsetreg.models.flow_matching import ConditionalVelocityField
from hepsetreg.models.heads import RegressionHead
from hepsetreg.models.tokenizer import ObjectGroupSpec

_KNOWN_TERMS = {"regression", "distribution_kl", "flow_matching", "consistency"}


def _resolve_callable(dotted_path: str):
    """Resolves e.g. ``"my_pkg.physics:ttbar_mass_consistency"`` (or the
    dotted form ``"my_pkg.physics.ttbar_mass_consistency"``) to the callable.
    Used for the user-supplied physics-consistency function, which cannot be
    expressed directly in YAML.
    """
    module_name, sep, attr_path = dotted_path.partition(":")
    if not sep:
        module_name, _, attr_path = dotted_path.rpartition(".")
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def build_groups(cfg: DictConfig) -> List[ObjectGroupSpec]:
    return [ObjectGroupSpec(name=str(g.name), in_features=int(g.in_features)) for g in cfg.model.groups]


def build_loss(cfg: DictConfig) -> CompositeLoss:
    terms: dict = {}
    configs: dict = {}

    for name, term_cfg in cfg.loss.terms.items():
        if name not in _KNOWN_TERMS:
            raise ValueError(f"Unknown loss term '{name}'. Expected one of {sorted(_KNOWN_TERMS)}.")

        configs[name] = LossTermConfig(
            weight=float(term_cfg.get("weight", 1.0)),
            ramp_epochs=int(term_cfg.get("ramp_epochs", 0)),
        )

        if name == "regression":
            terms[name] = RegressionLoss(kind=str(term_cfg.get("kind", "huber")), delta=float(term_cfg.get("delta", 1.0)))
        elif name == "distribution_kl":
            terms[name] = HistogramKLDivergenceLoss(
                bins=int(term_cfg.get("bins", 100)),
                sigma=float(term_cfg.get("sigma", 0.4)),
                eps=float(term_cfg.get("eps", 1e-8)),
                hist_min=term_cfg.get("hist_min", None),
                hist_max=term_cfg.get("hist_max", None),
                dynamic_padding=float(term_cfg.get("dynamic_padding", 0.25)),
            )
        elif name == "flow_matching":
            terms[name] = ConditionalFlowMatchingLoss()
        elif name == "consistency":
            fn_path = term_cfg.get("fn", None)
            if not fn_path:
                raise ValueError(
                    "loss.terms.consistency.fn must be a dotted path to a callable, "
                    "e.g. 'my_project.physics:ttbar_mass_consistency'."
                )
            terms[name] = PhysicsConsistencyLoss(_resolve_callable(str(fn_path)))

    return CompositeLoss(terms, configs)


def build_datamodule(cfg: DictConfig) -> RegressionDataModule:
    d = cfg.data
    return RegressionDataModule(
        group_names=[str(n) for n in d.group_names],
        train_file=str(d.train_file),
        val_file=str(d.val_file),
        test_file=str(d.test_file) if d.get("test_file", None) else None,
        target_key=str(d.get("target_key", "target")),
        extra_keys=[str(n) for n in (d.get("extra_keys", []) or [])],
        batch_size=int(d.get("batch_size", 256)),
        num_workers=int(d.get("num_workers", 4)),
    )


def build_target_scaler(cfg: DictConfig) -> Optional[TargetScaler]:
    scaler_file = cfg.data.get("scaler_file", None)
    if not scaler_file:
        return None
    return TargetScaler.from_hdf5(scaler_file)


def build_lightning_module(cfg: DictConfig):
    mode = str(cfg.model.get("mode", "supervised"))
    groups = build_groups(cfg)
    loss = build_loss(cfg)
    target_scaler = build_target_scaler(cfg)
    train_cfg: Any = OmegaConf.to_container(cfg.train, resolve=True)

    if mode == "supervised":
        encoder = ObjectSetEncoder(
            groups=groups,
            d_model=int(cfg.model.d_model),
            nhead=int(cfg.model.nhead),
            num_layers=int(cfg.model.num_layers),
            dim_feedforward=cfg.model.get("dim_feedforward", None),
            dropout=float(cfg.model.get("dropout", 0.1)),
            pooling=str(cfg.model.get("pooling", "cls")),
            use_type_embedding=bool(cfg.model.get("use_type_embedding", True)),
        )
        head_cfg = cfg.model.get("head", {})
        head = RegressionHead(
            d_model=int(cfg.model.d_model),
            output_dim=int(cfg.model.output_dim),
            n_layers=int(head_cfg.get("n_layers", 3)),
            start_neurons=int(head_cfg.get("start_neurons", 128)),
            dropout=float(head_cfg.get("dropout", 0.05)),
        )
        return SupervisedRegressor(encoder, head, loss, train_cfg=train_cfg, target_scaler=target_scaler)

    if mode == "flow_matching":
        velocity_field = ConditionalVelocityField(
            groups=groups,
            output_dim=int(cfg.model.output_dim),
            d_model=int(cfg.model.d_model),
            nhead=int(cfg.model.nhead),
            num_layers=int(cfg.model.num_layers),
            dim_feedforward=cfg.model.get("dim_feedforward", None),
            dropout=float(cfg.model.get("dropout", 0.1)),
            use_type_embedding=bool(cfg.model.get("use_type_embedding", True)),
        )
        sampling_raw = cfg.get("sampling", None)
        sampling_cfg = (
            SamplingConfig(**OmegaConf.to_container(sampling_raw, resolve=True)) if sampling_raw else SamplingConfig()
        )
        return FlowMatchingRegressor(
            velocity_field,
            loss,
            output_dim=int(cfg.model.output_dim),
            train_cfg=train_cfg,
            sampling_cfg=sampling_cfg,
            target_scaler=target_scaler,
        )

    raise ValueError(f"Unknown model.mode '{mode}'. Expected 'supervised' or 'flow_matching'.")
