#!/usr/bin/env bash
# Sets up everything needed to run the basic regression model:
# installs uv, downloads a test data file, installs the Python environment,
# then proves the whole chain actually works by loading the downloaded file
# through the project's own code.
#
# Usage (from a fresh git clone):
#   cd HepSetReg
#   ./prep_bsc.sh

set -euo pipefail
trap 'echo "prep_bsc.sh failed at line $LINENO -- see the message above for which step this was." >&2' ERR

DATA_URL="https://cernbox.cern.ch/s/EBakp1JpfphYRcM/download"
DATA_DIR="datasets"
DATA_FILE="$DATA_DIR/tt2Ltest.parquet"
MIN_DATA_BYTES=1000000  # a truncated/failed download would be far smaller than this

# ---- sanity check: are we in the right place? ----
# (the most likely real mistake: running this before `cd`-ing into the clone)
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: no pyproject.toml here. Run this script from inside the" >&2
    echo "HepSetReg directory you just cloned (cd HepSetReg first)." >&2
    exit 1
fi

# ---- step 1: install uv (skip if already installed) ----
echo "== Step 1: uv =="
if command -v uv &> /dev/null; then
    echo "uv already installed ($(command -v uv)), skipping install."
else
    if ! command -v curl &> /dev/null; then
        echo "ERROR: curl is not installed. Install curl first, then re-run this script." >&2
        exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # the installer updates PATH for future shells (via .bashrc etc.), not this
    # one -- add its usual install locations so the rest of this script can
    # find it without starting a new shell.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv install finished but 'uv' still isn't on PATH." >&2
    echo "Open a new terminal (so your shell picks up the updated PATH) and re-run this script." >&2
    exit 1
fi
echo "uv OK: $(uv --version)"

# ---- step 2: download the test data (skip if already present) ----
echo "== Step 2: test data =="
mkdir -p "$DATA_DIR"

if [ -f "$DATA_FILE" ] && [ "$(stat -c%s "$DATA_FILE" 2>/dev/null || stat -f%z "$DATA_FILE")" -ge "$MIN_DATA_BYTES" ]; then
    echo "$DATA_FILE already downloaded, skipping."
else
    echo "Downloading test data to $DATA_FILE ..."
    # --fail: treat HTTP error responses (404, 403, ...) as failures instead of
    # silently saving the error page as if it were the data file.
    # --retry: tolerate a flaky connection.
    if ! curl --fail --retry 3 -L -o "$DATA_FILE" "$DATA_URL"; then
        echo "ERROR: download failed. Check your network connection and that" >&2
        echo "$DATA_URL is still a valid link." >&2
        rm -f "$DATA_FILE"
        exit 1
    fi
fi

actual_bytes=$(stat -c%s "$DATA_FILE" 2>/dev/null || stat -f%z "$DATA_FILE")
if [ "$actual_bytes" -lt "$MIN_DATA_BYTES" ]; then
    echo "ERROR: $DATA_FILE is only $actual_bytes bytes -- looks like a failed" >&2
    echo "or truncated download, not real data (re-run this script to retry)." >&2
    rm -f "$DATA_FILE"
    exit 1
fi
echo "Data file OK: $DATA_FILE ($actual_bytes bytes)"

# ---- step 3: install the Python environment ----
# --extra dev: also installs ipykernel/jupyter/matplotlib/pytest, since
# students will want notebooks and plotting, not just the bare training
# scripts. Plain `uv sync` *removes* those packages if anything had already
# installed them, rather than just skipping them -- always use --extra dev here.
echo "== Step 3: uv sync =="
if ! uv sync --extra dev; then
    echo "ERROR: 'uv sync' failed -- see the error above. A slow/interrupted" >&2
    echo "network is the most common cause; just re-run this script." >&2
    exit 1
fi

# ---- step 4: prove the whole chain actually works ----
# Loads the downloaded file through the project's own dataset-loading code,
# using the same field names configs/run_regression.yaml uses -- this checks
# not just "is the environment installed" but "can the package actually read
# real data", which is the thing that's actually going to be run.
echo "== Step 4: test run =="
uv run python - "$DATA_FILE" <<'PY'
import sys
import yaml

data_file = sys.argv[1]

from hepsetreg.datasets.EventDatasetBase import load_arrays

with open("configs/run_regression.yaml") as f:
    features = yaml.safe_load(f)["features"]

global_arr, jet_arr, lepton_arr, target_arr = load_arrays(data_file, features)

print(f"Loaded {len(global_arr)} events from {data_file}")
print(f"  global features: {global_arr.shape[1]}")
print(f"  mean {features['target'][0]}: {target_arr[:, 0].mean():.1f}")
PY

echo
echo "All done. The environment is set up and the test data loads correctly."
echo "See docs/bsc_instructions.md for how to train and predict."
