# sign_ml

Full machine learning pipeline for traffic sign classification.

## Project description
The goal of the project is to classify traffic signs using machine learning.
Initially, the project will use a simple kaggle dataset with around 6000 rgb images of 52 different types summing up to around 200MB. The images vary in size and are cropped to the main traffic sign.
https://www.kaggle.com/datasets/tuanai/traffic-signs-dataset/data
We will be training a CNN from scratch and might also try deploying a pre-trained model, fine-tuning it for traffic sign classification.

## Project structure

The directory structure of the project looks like this:
```txt
├── 1README.md                # Exam checklist / template
├── AGENTS.md                 # Instructions for autonomous coding agents
├── command.md                # CLI usage examples
├── data.dvc                  # DVC tracking for data/ directory
├── configs/                  # Hydra configuration files
│   ├── config.yaml
│   ├── sweep.yaml
│   ├── tensorboardprofiling.yaml
│   └── experiment/
│       ├── exp1.yaml
│       └── exp2.yaml
├── data/
│   ├── processed/            # Preprocessed tensors (.pt) used by training
│   └── raw/                  # Original traffic signs data and metadata
├── dockerfiles/              # Dockerfiles for training and API
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/                     # MkDocs documentation
│   ├── mkdocs.yaml
│   ├── README.md
│   └── source/
│       └── index.md
├── log/                      # Logs (e.g. API jobs)
├── models/                   # Saved model weights
├── notebooks/                # Exploratory notebooks
├── outputs/                  # Training run outputs (per date / timestamp)
├── reports/
│   └── figures/              # Plots and figures
├── src/
│   └── sign_ml/              # Package with project source code
│       ├── __init__.py
│       ├── api.py            # FastAPI inference service
│       ├── data.py           # Data loading and preprocessing
│       ├── data_distributed.py
│       ├── evaluate.py
│       ├── merge_data.py
│       ├── model.py          # CNN model definition
│       ├── train.py          # Training entry point (DDP‑ready)
│       ├── utils.py
│       └── visualize.py
├── tests/                    # Unit, integration and performance tests
│   ├── integrationtests/
│   │   └── test_api.py
│   ├── performancetests/
│   │   └── test_locustfile.py
│   └── unittests/
│       ├── __init__.py
│       ├── test_data.py
│       ├── test_data_distributed.py
│       └── test_model.py
├── LICENSE
├── pyproject.toml            # Project metadata and dependencies
├── README.md                 # This file
├── ruff.toml                 # Ruff lint/format configuration
└── tasks.py                  # Invoke tasks (lint, test, etc.)
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
