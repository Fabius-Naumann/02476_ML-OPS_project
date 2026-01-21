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

##  c-profiling (profiler for data.py, train.py, evaluate.py)

Run from inside `src/sign_ml/`:

```bash
py -m cProfile -o profile_data.prof -s cumtime data.py
python -c "import pstats; pstats.Stats('profile_data.prof').sort_stats('cumulative').print_stats(40)"

py -m cProfile -o profile_train.prof -s cumtime train.py
py -c "import pstats; pstats.Stats('profile_train.prof').sort_stats('cumulative').print_stats(40)"

py -m cProfile -o profile_evaluate.prof -s cumtime evaluate.py
py -c "import pstats; pstats.Stats('profile_evaluate.prof').sort_stats('cumulative').print_stats(40)"
```

### PyTorch profiler (train.py, evaluate.py)

Your `train.py` and `evaluate.py` support an optional `torch.profiler` mode. It profiles only a small number of
steps (default: 10).

`log/<timestamp>/trace.json`

Run from inside root:

```bash
# Use the dedicated config that enables TensorBoard profiling output under project-root ./log/
#Run from inside root

py -m sign_ml.train --config-name tensorboardprofiling

# `evaluate.py` creates TensorBoard profiler traces under project-root ./log/ by default
#Run from root
py -m sign_ml.evaluate --config-name tensorboardprofiling

```


Run from the project root:
```bash
tensorboard --logdir=./log
```
Then start TensorBoard (from the project root) and open <http://localhost:6006/#pytorch_profiler>:

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

### unit tests (with pip)

Use this to verify that the package can be imported and the unit tests pass after dependency changes.

Run from the project root:
```bash
python -m pip install -e .
python -m pytest -q tests/
```
or
```bash
# Install the package itself
pip install -e .

# Install missing test deps explicitly (pip doesn’t read uv dev groups)
pip install httpx pytest coverage

# Run tests with coverage
python -m coverage run -m pytest tests/
python -m coverage report
```

## FastAPI (inference API)

FastAPI is a lightweight Python web framework for building REST APIs.
In this project it is used to expose the trained model as an HTTP service, so other programs (or users) can send an
image and receive a prediction without running Python code directly.

### How to run exactly like your command (from `src/sign_ml`)

If your current working directory is `src/sign_ml/`:

```bash
python -m uvicorn --reload --port 8000 api:app
```

Tip: Prefer `python` (your active environment) over the Windows `py` launcher to avoid running uvicorn in a different
Python installation.

Then open:

- <http://localhost:8000>
- <http://localhost:8000/docs>

### Endpoints

Your inference API is defined in `src/sign_ml/api.py` and exposes endpoints like:

This service exposes a FastAPI application for health checks, basic model metadata, inference, and admin utilities.

### Meta
- GET `/`
  Returns a short usage hint.

- GET `/health`
  Returns service health and model load status.
  - Response: `{ status: "ok"|"not_ready", is_loaded: bool, weights_file: str, num_classes: int|null, detail: str|null }`

- GET `/model`
  Returns basic model metadata.
  - Response: `{ model_path: str, num_classes: int|null, device: str }`

### Inference
- POST `/predict`
  Runs inference on a single uploaded image.
  - Request: multipart/form-data with key `image` (content-type `image/*`).
  - Responses:
    - 200: `{ predicted_class: int, probabilities: number[], num_classes: int }`
    - 400: invalid image or empty file
    - 415: non-image media type
    - 503: model not loaded

### Admin
Admin endpoints run jobs in a controlled subprocess and return latest status/results.

- GET `/admin/status`
  Indicates whether admin endpoints are enabled and current job counts.
  - Response: `{ enabled: bool, max_running_jobs: int, running_jobs: int, total_jobs: int }`

- GET `/admin/train`
  Returns info about the latest train job (does not start a new job).
  - Response: `{ action: "train", status: "not_started"|"running"|"completed"|"failed"|"from_outputs", job_id: str|null, log_tail: string[] }`

- POST `/admin/train_sync`
  Starts training and waits up to a configured timeout before returning the result/status.
  - Query params (optional): `epochs: int`, `batch_size: int`, `lr: float`
  - Response (success): `{ job_id: str, action: "train", status: "completed"|"running"|"failed", return_code: int|null, log_tail: string[] }`

- GET `/admin/evaluate`
  Returns info about the latest evaluate job (does not start a new job).
  - Response: `{ action: "evaluate", status: "not_started"|"running"|"completed"|"failed"|"from_outputs", job_id: str|null, log_tail: string[] }`

- POST `/admin/evaluate_sync`
  Starts evaluation and waits up to a configured timeout before returning the result/status.
  - Query params (optional): `batch_size: int`
  - Response (success): `{ job_id: str, action: "evaluate", status: "completed"|"running"|"failed", return_code: int|null, log_tail: string[] }`

- POST `/admin/test_sync`
  Runs the repository test suite and returns the result/status.
  - Response (success): `{ job_id: str, action: "test", status: "completed"|"running"|"failed", return_code: int|null, log_tail: string[] }`

---

# API test (integration testing) for FastAPI

API testing validates the application programming interface (API) directly for correctness, reliability, and security.
Unlike unit tests, it tests the API as a whole (integration testing) rather than individual functions.
The tests should simulate realistic user requests (paths, methods, headers, and payloads).

These tests validate API behavior end-to-end (in-process) using FastAPI's `TestClient`.
They do **not** require a running `uvicorn` server.

Run from the project root:


```bash
# Fallback (pip)
python -m pip install -e .
python -m pip install httpx
#Remember to add httpx to your requirements.txt or pyproject.toml file

# run the tests from the project root
python -m pytest -q tests/integrationtests/test_api.py
# for more verbose output: from the project root
python -m pytest -vv tests/integrationtests/test_api.py
```

---

## Load testing (Locust)

Load testing measures how the API behaves under normal and peak conditions, helping identify bottlenecks and capacity.
We use the Python tool [Locust](https://locust.io/) to simulate concurrent users hitting the core endpoints.

our Locust script at
`tests/performancetests/test_locustfile.py`.

### Install (pip)

Run from the project root:
```bash
pip install locust
# #Remember to add httpx to your requirements.txt or pyproject.toml file
```

If the `locust` command is not on PATH, use the module form:
```bash
python -m locust --version
```

### Start the API (pip)

Run the inference API locally:
Run from  `src/sign_ml/`:

```bash
python -m uvicorn --reload --port 8000 api:app
```

### Run Locust (Web UI) from root

```bash
locust -f tests/performancetests/test_locustfile.py
```
Open <http://localhost:8089>, and set `Host` to <http://localhost:8000> (or your deployed URL). Then click Start.
```
