"""Thin YAML config loading (OmegaConf `DictConfig`, with dot-access like
DIRECTOR's/ReconstructionAndSB's ``cfg.model.d_model`` style) plus optional
dotlist overrides (``train.max_epochs=5``) -- Hydra-lite, without pulling in
Hydra's multirun/output-directory machinery for what is meant to stay a small
package.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(path: Union[str, Path], overrides: Optional[List[str]] = None) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg
