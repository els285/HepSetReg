# HepSetReg for BSc projects

This branch is for BSc projects hosted at University of Manchester in Autumn 2026.

> [!CAUTION]
> Commands in this notebook are specific to the University of Manchester CSF3 cluster and will not run elsewhere.

# Instructions for logging in to CSF3

> [!IMPORTANT]
> You will need to follow this procedure every time you start a new session

We will run inside the CSF terminal. A terminal is a way of ineracting with the computer's operating system through tetx-based commands.
Open VSCode and click `Terminal` in the top dropdown menu, then click `New Terminal`.

We will `ssh` into the CSF3 cluster. SSH is simply a way to access a remote computer or server given an internet connection. 
In the terminal you should be able to type commands using the keyboard. Type the following command, replacing `username` with your University of Manchester username.
```
ssh username@csf3.itservices.manchester.ac.uk
```
Press ENTER.
If this works, you should be prompted for your UoM password. Enter it and press ENTER.
If your password is correct, you should then be prompted for two-factor authentication. For me this means I have to press 1 and then ENTER, and then use the DUO two-factor authentication app to confirm.
If all this works, you should be in!

If you wish to use this when not on campus wifi, you need a special access through something called a  VPN. See this website to set up: https://www.itservices.manchester.ac.uk/ourservices/vpn/


## 🟦 1. Set up the environment

> [!NOTE]
> You will only need to follow these instructions the first time you do this

Clone the repository and switch into it:
```bash
git clone -b bsc https://github.com/els285/HepSetReg.git
cd HepSetReg
```

Run the preparation script:
```bash
./setup_bsc
```

This installs [uv](https://docs.astral.sh/uv/) if you don't have it,
downloads a small test data file to `datasets/`, installs the Python
environment, and then does a test run to prove the whole chain actually
works. You only need to run this once.

If it fails partway through, read the error message it prints -- it's
written to tell you what went wrong and what to do about it -- then just
re-run `./setup_bsc`; each step skips itself if it's already done.

Everything below assumes you're running commands from the repo root (the
directory `pyproject.toml` and `configs/` are in) -- the paths in the
config file are plain relative paths, so they only resolve correctly from
there.

**Training data:** `setup_bsc` only downloads a small test-split file, to
check your environment works -- it's not enough to train on by itself.
Ask your instructor where to get the full train/val parquet files that
`configs/run_regression.yaml`'s `train_path`/`val_path` expect, and put
them in the repo root (or edit those two lines to point at wherever you
put them).

## 🟩 2. Look at the config

> [!TIP]
> To open, read and edit files, you can use the Remote-SSH package in VSCode.
> Alternatively, you can use `nano` to read and edit
> * `nano <file>` to open the file
> * Edit the file by navigating around with arrow keys and using keyboard
> * `Ctrl`+`O` to save the file: press `ENTER` to confirm

Open `configs/run_regression.yaml`. It has three sections:

- `features:` -- which columns to read from the parquet file. You shouldn't
  need to touch this.
- `train:` -- data paths, model size, and training settings.
- `predict:` -- where to find the checkpoint and what to run inference on,
  filled in once you've trained something (step 4).

If you want to experiment with hyperparameters (`learning_rate`,
`max_epochs`, `d_model`, ...), edit the `train:` section directly, or make a
copy of the file and point `--config` at your copy instead.

## 🟨 3. Run model training interactively
When we ssh in to CSF, we are in a login node that does not have direct access to a GPU. 
From this node, we must tunnel into a worker node with a GPU. Run
```
./get_interactive_gpu
```
to access a GPU node. 
This can sometimes take a long time depending on how busy the cluster is; for this reason we can also use a service that runs things automatically for us - see `Running on cluster` section below.

When on the GPU node, you train a model via:
```bash
./train
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

## 🟧 4. Fill in the predict section

Open `configs/run_regression.yaml` again and look inside the folder that got
printed in step 3 (e.g. `tb_logs/event_regression/version_0/checkpoints/`).
Lightning names the checkpoint file `epoch=X-step=Y.ckpt`. Update the
`predict:` section's `checkpoint:` and `stats:` paths to match exactly what
you see there -- `version_0` in the example above may be a different number
for you.

## 🟥 5. Predict

```bash
./predict
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

## If something goes wrong

- **`./setup_bsc` fails** -- it prints what step failed and why; the most
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
