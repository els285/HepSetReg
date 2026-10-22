"""
Tiny YAML config loader shared by every run_*.py script.

Each run script still has exactly the same `train(config)` / `predict(config)`
functions as before, each expecting a plain dict -- the only thing that
changes is where that dict comes from. `load_config(stage)` reads the path
given after `--config` on the command line and returns that section
("train" or "predict") of the YAML file, with any trailing `key=value`
arguments applied as overrides on top.
"""

import sys

import yaml


def check_args(usage):
    """Call this BEFORE the heavy imports (torch, pytorch_lightning, ...) at
    the top of each run_*.py's `if __name__ == "__main__":` guard. A bad
    command line (missing --train/--predict or --config) then fails in
    milliseconds, instead of only being reported after paying the full
    import cost -- which on a slow filesystem can be tens of seconds."""
    has_mode = "--train" in sys.argv or "--predict" in sys.argv
    has_config = "--config" in sys.argv
    if not (has_mode and has_config):
        print(usage)
        sys.exit(1)


def load_config(stage):
    if "--config" not in sys.argv:
        raise SystemExit("missing --config <path.yaml>")
    path = sys.argv[sys.argv.index("--config") + 1]
    with open(path) as f:
        full_config = yaml.safe_load(f)

    config = full_config[stage]
    # anything outside train:/predict: (feature/branch names, aim_repo, ...)
    # is shared by both stages, so merge it into whichever one was asked for.
    for key, value in full_config.items():
        if key not in ("train", "predict"):
            config[key] = value

    # any remaining bare `key=value` arguments override entries in the config,
    # e.g. `./train run_regression.yaml learning_rate=0.0005 max_epochs=5`.
    # yaml.safe_load coerces the value string to the right type (float, int,
    # bool, ...) the same way the YAML file itself would.
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("-"):
            key, _, value = arg.partition("=")
            config[key] = yaml.safe_load(value)

    return config
