## Hyperparameter optimization (W&B sweeps)

Run these from the  root project (so `configs/sweep.yaml` resolves correctly):

```bash
# One-time per machine/session (as needed)
wandb login

# Create the sweep
wandb sweep configs/sweep.yaml

# Start the sweep agent
# Use the command printed by the sweep creation, e.g.:
py -m wandb agent <entity>/<project>/<sweep_id>
```

# Profilers

---

!!! info "Core Module"

## Project profiling (sign_ml)

Run from inside `src/sign_ml/`:

```bash
py -m cProfile -o profile_data.prof -s cumtime data.py
python -c "import pstats; pstats.Stats('profile_data.prof').sort_stats('cumulative').print_stats(40)"

py -m cProfile -o profile_train.prof -s cumtime train.py
py -c "import pstats; pstats.Stats('profile_train.prof').sort_stats('cumulative').print_stats(40)"

py -m cProfile -o profile_evaluate.prof -s cumtime evaluate.py
py -c "import pstats; pstats.Stats('profile_evaluate.prof').sort_stats('cumulative').print_stats(40)"
```

### PyTorch profiler (your code)

Your `train.py` and `evaluate.py` support an optional `torch.profiler` mode. It profiles only a small number of
steps (default: 10) and writes a Chrome trace to:

`log/sign_ml/profiling/torch/<timestamp>/trace.json`

Run from inside root:

```bash
# Use the dedicated config that enables TensorBoard profiling output under project-root ./log/
#Run from inside root

py -m sign_ml.train --config-name tensorboardprofiling

# `evaluate.py` creates TensorBoard profiler traces under project-root ./log/ by default
#Run from root
py -m sign_ml.evaluate --config-name tensorboardprofiling
# Alternative (explicit flags)
# python train.py +profiling.torch.enabled=true +profiling.torch.export_tensorboard=true
# python evaluate.py +profiling.torch.enabled=true +profiling.torch.export_tensorboard=true
```


Run from the project root:
```bash
tensorboard --logdir=./log
```
Then start TensorBoard (from the project root) and open <http://localhost:6006/#pytorch_profiler>:


Alternative: keep `+profiling.torch.export_chrome=true` (and disable TensorBoard export) to generate a `trace.json`,
then open it via `chrome://tracing`.

---

## Development tooling (pre-commit, ruff, mypy)

We use `pre-commit` to run formatting, linting, type-checking, and `uv lock` checks.
Prefer the `uv` workflow if you have it available; use the pip fallback only if you don't.

### Recommended (uv)

```bash
# Install uv (if you don't have it)
python -m pip install -U uv

# Install pre-commit and hook dependencies (managed via uv)
uv run pre-commit install
uv run pre-commit run --all-files
```

### Fallback (pip, without uv)

What this does:
- Installs the tooling (Ruff = formatting/linting, mypy = type-checking) and sets up the `pre-commit` git hook.
- Runs all hooks on the entire repo, so formatting/lint/type issues are caught locally before CI.

Tip: Installing the Ruff VS Code extension helps catch issues while you edit.

Use this if you don't have `uv` available and want to use plain `pip`.

Run from the project root:
```bash
python -m pip install -U pre-commit ruff mypy
python -m pre_commit install
python -m pre_commit run --all-files
```

### Quick verification (pip)

Use this to verify that the package can be imported and the unit tests pass after dependency changes.

Run from the project root:
```bash
python -m pip install -e .
python -m pytest -q tests/
```
