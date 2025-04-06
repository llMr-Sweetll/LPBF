# LPBF Optimizer

A physics-informed, AI-driven optimizer for Laser Powder Bed Fusion manufacturing processes.

## Overview

This project implements a physics-informed neural network (PINN) that acts as a surrogate model for predicting key outcomes of the LPBF process (residual stress, porosity, geometric accuracy) based on process parameters (laser power, scan speed, etc.). The PINN is then used in a multi-objective optimizer to find optimal scan vectors that balance multiple competing objectives.

## Big-picture flow

```
 ┌──────────┐   FE-sim data   ┌───────────────┐   labelled tensors   ┌──────────┐
 │  FEA /   │ ──────────────► │ Physics-      │ ────────────────►    │  Multi-  │
 │  CAE     │  + in-situ      │ informed NN   │  (σ, φ, GAR, etc.)   │ objective│
 │  models  │   frames        │  (PINN)       │                      │ optimiser│
 └──────────┘                 └───────────────┘                       └──────────┘
      ▲                               │                                     │
      │      new scan vector S* ◄─────┴────────── Pareto set ───────────────┘
      │
   build coupons & feed back (validation loop)
```

## Mathematical modeling

The PINN is built on the following key equations:

### Heat equation with moving laser

```
ρc_p∂T/∂t = ∇·(k∇T) + 2ηP/(πr_0²)exp(-2r²/r_0²) - H_m∂f_s/∂t
```

### Static equilibrium for residual stress (elastic-viscoplastic)

```
∇·σ = 0,   σ = C:ε^e,   ε̇^p = A(σ_eq/σ_y)^n
```

## Project Structure

```
lpbf-optimizer/
├── data/
│   ├── raw/               <- FEA outputs, synchrotron frames
│   ├── processed/         <- tensors ready for NN
│   └── params.yaml        <- Configuration parameters
├── src/
│   ├── fea_runner.py      <- wraps ABAQUS / COMSOL jobs
│   ├── preprocessing.py   <- convert .odb/.vtk to torch tensors
│   ├── pinn/
│   │   ├── model.py       <- PINN definition (PyTorch)
│   │   ├── physics.py     <- PDE residuals
│   │   └── train.py       <- Training loop
│   ├── optimiser/
│   │   ├── nsga3.py       <- genetic algorithm (pymoo)
│   │   └── bayesopt.py    <- alternative (Ax / BoTorch)
│   └── validate/
│       ├── build_runner.py<- sends G-code to LPBF machine
│       └── characterise.py<- XCT, EBSD parsers
├── notebooks/             <- quick EDA & plotting
├── tests/                 <- unit tests (pytest)
└── README.md
```

## Core Technologies

- **Language**: Python 3.11
- **Libraries**: PyTorch, torchdiffeq, pymoo, numpy, scipy, matplotlib, h5py

## Implementation Plan

| Month | Milestone | Code deliverable |
|-------|-----------|------------------|
| 1-2 | Set up repo, CI/CD, write `fea_runner.py` | `data/raw/*.h5` |
| 3-4 | Finish FEA mesh scripts, generate 1 TB dataset | `preprocessing.py` |
| 5-6 | Draft PINN, unit tests pass (R² > 0.8) | `pinn/model.py` |
| 7-9 | Hyper-param tuning, add physics loss | `train.py` checkpoints |
| 10-12 | Integrate NSGA-III optimiser | `optimiser/nsga3.py` |
| 13-15 | Transfer-learning to IN718 | new `props.yaml` |
| 16-18 | Build coupons, write `validate/` parsers | validation plots |
| 19-21 | Feedback loop: retrain with experimental labels | v2.0 checkpoint |
| 22-24 | Final Pareto map, manuscript notebooks | `notebooks/final.ipynb` |