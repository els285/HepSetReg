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

**Training data:** `configs/run_regression.yaml`'s `train_path`/`val_path`/
`data_path` all point at the small test file `setup_bsc` downloads for now,
so you can run everything below right away -- it's too small to produce a
good model, just enough to prove the whole chain works. Ask your instructor
for the full train/val parquet files when you're ready to train for real,
and edit those lines to point at wherever you put them.

## 🟩 2. Run model training interactively
When we ssh in to CSF, we are in a login node that does not have direct access to a GPU. 
From this node, we must tunnel into a worker node with a GPU. Run
```
./get_interactive_gpu
```
to access a GPU node. 
This can sometimes take a long time depending on how busy the cluster is; for this reason we can also use a service that runs things automatically for us - see `Running on cluster` section below.

When on the GPU node, you train a model by:
```bash
./train run_regression.yaml
```
where `run_regression.yaml`  is the name of the configuration file inside `configs/`.
This config contains all the information the code needs to set up the training, including the names of the input data files, the "hyperparameters" of the network itself etc.

You can change these setings by adding `key=value` after
the config name when running the `train` command:
```bash
./train run_regression.yaml max_epochs=100
```

This only overrides that one setting for this one run -- the config file
itself is unchanged. Works the same way with `./predict`. (See step 4 below
if you'd rather change a setting permanently instead.)

Each run gets its own folder under `tb_logs/event_regression/`, named
`version_0`, `version_1`, ... by default. Give it a name you'll recognise
instead with `run_name=`:
```bash
./train run_regression.yaml run_name=my_first_run
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

## 🟨 3. Predict

Look inside the folder that just got printed (e.g.
`tb_logs/event_regression/version_0/checkpoints/`) to find the checkpoint
file -- Lightning names it `epoch=X-step=Y.ckpt`. Pass `checkpoint=`/`stats=`
relative to that `version_0`/`run_name` folder -- `./predict` already knows
about the `tb_logs/event_regression/` part:

```bash
./predict run_regression.yaml checkpoint=version_0/checkpoints/epoch=X-step=Y.ckpt stats=version_0/stats.pt
```

(`version_0` and `epoch=X-step=Y` are whatever you actually saw printed and
found in that folder -- they're different for every run, or whatever you
set `run_name=` to in step 2.)

This prints the mean absolute error on the test set, and writes predictions
alongside the true values to the parquet file named in `output:` -- add
`output=some_name.parquet` the same way if you want a different output
filename.

## 🟧 4. Editing the config file

> [!TIP]
> To open, read and edit files, you can use the Remote-SSH package in VSCode.
> Alternatively, you can use `nano` to read and edit
> * `nano <file>` to open the file
> * Edit the file by navigating around with arrow keys and using keyboard
> * `Ctrl`+`O` to save the file: press `ENTER` to confirm

The `key=value` overrides above only apply to one run at a time. To change a
setting permanently -- or just to see everything that's there -- open
`configs/run_regression.yaml` itself. It has three sections:

- `features:` -- which columns to read from the parquet file. You shouldn't
  need to touch this.
- `train:` -- data paths, model size, and training settings.
- `predict:` -- where to find the checkpoint and what to run inference on.
  You don't need to fill this in by hand (step 3's overrides cover it), but
  it's a useful record of your last run, or a place to put defaults so you
  don't have to type the paths out every time. Unlike the `checkpoint=`/
  `stats=` overrides in step 3, `checkpoint:`/`stats:` written here need the
  full path (`tb_logs/event_regression/version_0/...`) -- only `./predict`'s
  command-line overrides get the `tb_logs/event_regression/` part added
  automatically.

If you want to experiment with hyperparameters (`learning_rate`,
`max_epochs`, `d_model`, ...), edit the `train:` section directly, or make a
copy of the file and point `--config` at your copy instead.

## Running on the cluster instead of interactively

`submit_run_regression` is a ready-to-use Slurm submit script -- it runs
exactly the training command from step 2. Submit it with:

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
- **Predict fails with a missing checkpoint/stats file** -- double check the
  `checkpoint=`/`stats=` paths you passed match the `version_N` folder and
  filename exactly.

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
