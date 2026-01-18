## Hyperparameter optimization (W&B sweeps)

Run these from the  root project (so `configs/sweep.yaml` resolves correctly):

```bash
# One-time per machine/session (as needed)
wandb login

# Create the sweep
wandb sweep configs/sweep.yaml

# Start the sweep agent
# Use the command printed by the sweep creation, e.g.:
#   wandb agent <entity>/<project>/<sweep_id>
wandb agent <sweep_id>
```

# Profilers

---

!!! info "Core Module"

## Project profiling (sign_ml)

Run from inside `src/sign_ml/`:

```bash
python -m cProfile -o profile_data.prof -s cumtime data.py
python -c "import pstats; pstats.Stats('profile_data.prof').sort_stats('cumulative').print_stats(40)"

python -m cProfile -o profile_train.prof -s cumtime train.py
python -c "import pstats; pstats.Stats('profile_train.prof').sort_stats('cumulative').print_stats(40)"

python -m cProfile -o profile_evaluate.prof -s cumtime evaluate.py
python -c "import pstats; pstats.Stats('profile_evaluate.prof').sort_stats('cumulative').print_stats(40)"
```

### PyTorch profiler (your code)

Your `train.py` and `evaluate.py` support an optional `torch.profiler` mode. It profiles only a small number of
steps (default: 10) and writes a Chrome trace to:

`log/sign_ml/profiling/torch/<timestamp>/trace.json`

Run from inside `src/sign_ml/`:

```bash
# Use the dedicated config that enables TensorBoard profiling output under project-root ./log/
#Run from inside `src/sign_ml/`

python train.py --config-name tensorboardprofiling

# `evaluate.py` creates TensorBoard profiler traces under project-root ./log/ by default
#Run from inside `src/sign_ml/`
python evaluate.py --config-name tensorboardprofiling

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
