# hepsetreg (bsc branch)

Predicts the ttbar invariant mass (`ttbar_mass`) from the reconstructed
jets, leptons and MET of a ttbar dilepton event, using a transformer that
reads the variable-length set of objects in each event.

This branch has one script (`run_regression.py`) and one config file
(`configs/run_regression.yaml`), so there's nothing to choose between --
just follow the steps below. (The full project, with several other
model/task combinations, is on the `config` branch.)

## 1. Set up the environment

From the repo root, after cloning:

```bash
./prep_bsc.sh
```

This installs [uv](https://docs.astral.sh/uv/) if you don't have it,
downloads a small test data file to `datasets/`, installs the Python
environment, and then does a test run to prove the whole chain actually
works. You only need to run this once.

If it fails partway through, read the error message it prints -- it's
written to tell you what went wrong and what to do about it -- then just
re-run `./prep_bsc.sh`; each step skips itself if it's already done.

Everything below assumes you're running commands from the repo root (the
directory `pyproject.toml` and `configs/` are in) -- the paths in the
config file are plain relative paths, so they only resolve correctly from
there.

**Training data:** `prep_bsc.sh` only downloads a small test-split file, to
check your environment works -- it's not enough to train on by itself.
Ask your instructor where to get the full train/val parquet files that
`configs/run_regression.yaml`'s `train_path`/`val_path` expect, and put
them in the repo root (or edit those two lines to point at wherever you
put them).

## 2. Look at the config

Open `configs/run_regression.yaml`. It has three sections:

- `features:` -- which columns to read from the parquet file. You shouldn't
  need to touch this.
- `train:` -- data paths, model size, and training settings.
- `predict:` -- where to find the checkpoint and what to run inference on,
  filled in once you've trained something (step 4).

If you want to experiment with hyperparameters (`learning_rate`,
`max_epochs`, `d_model`, ...), edit the `train:` section directly, or make a
copy of the file and point `--config` at your copy instead.

## 3. Train

```bash
uv run python -m hepsetreg.runs.run_regression --train --config configs/run_regression.yaml
```

This will print a line like:

```
done training. checkpoint + stats saved under tb_logs/event_regression/version_0
```

Watch it train (in a separate terminal, also from the repo root):

```bash
uv run tensorboard --logdir tb_logs
```

then open the URL it prints in a browser.

## 4. Fill in the predict section

Open `configs/run_regression.yaml` again and look inside the folder that got
printed in step 3 (e.g. `tb_logs/event_regression/version_0/checkpoints/`).
Lightning names the checkpoint file `epoch=X-step=Y.ckpt`. Update the
`predict:` section's `checkpoint:` and `stats:` paths to match exactly what
you see there -- `version_0` in the example above may be a different number
for you.

## 5. Predict

```bash
uv run python -m hepsetreg.runs.run_regression --predict --config configs/run_regression.yaml
```

This prints the mean absolute error on the test set, and writes predictions
alongside the true values to the parquet file named in `output:`.

## Running on the cluster instead of interactively

`submit_run_regression` is a ready-to-use Slurm submit script -- it runs
exactly the training command from step 3. Submit it with:

```bash
sbatch submit_run_regression
```

Check the `#SBATCH` lines at the top (partition, memory, wall time) match
what you've been told to use before submitting.

### Interactive GPU session

<!-- fill in: how to request an interactive GPU node on this cluster -->

## If something goes wrong

- **`./prep_bsc.sh` fails** -- it prints what step failed and why; the most
  common causes are no network connection, or running it from the wrong
  directory.
- **"usage: ... --config <path.yaml>"** -- you forgot `--config
  configs/run_regression.yaml` on the command line, or mistyped `--train`/
  `--predict`.
- **A file-not-found error on a `.parquet` file** -- you're not running from
  the repo root, or the data files aren't where `configs/run_regression.yaml`
  expects them (see the training-data note in step 1).
- **Predict fails with a missing checkpoint/stats file** -- you haven't done
  step 4 yet, or the `version_N` number is wrong.

## Looking at the code

`src/hepsetreg/runs/run_regression.py` is the script itself: a
`build_model`, a `train`, and a `predict` function, nothing else. It's built
from a few pieces elsewhere in `src/hepsetreg/`:

- `datasets/EventDatasetBase.py` -- reads the parquet file and pads it into
  fixed-size batches.
- `backbones/backbone.py` + `backbones/head.py` -- the transformer encoder
  and the regression head on top of it.
- `training/lightning_module.py` -- wires the model and loss into a
  PyTorch Lightning training loop.
