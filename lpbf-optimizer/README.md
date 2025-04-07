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

## Installation and Setup

### System Requirements

- Python 3.11 or later
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM (for large datasets)
- 100GB+ storage space (for datasets and model checkpoints)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lpbf-optimizer.git
cd lpbf-optimizer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) If using commercial FEA solvers, ensure they are properly installed and configured:
   - ABAQUS: Make sure the `abaqus` command is in your PATH
   - COMSOL: Configure paths in `data/params.yaml`

## Complete Workflow

The LPBF optimization workflow consists of several sequential steps:

### 1. Generate or Process Data

You can either:
- Generate synthetic data for development/testing
- Process real FEA simulation results
- Use a combination of both

#### 1.1 Generating Synthetic Data

```bash
python src/generate_synthetic_data.py --config data/params.yaml --scan-vectors 100 --points-per-vector 1000
```

This creates a dataset in `data/processed/lpbf_dataset.h5` with synthetic simulation data.

The synthetic data generation module simulates:
- Physical relationships between process parameters and outcomes
- Spatial variations in material properties
- Realistic noise and variations found in real-world measurements

Key parameters for synthetic data generation:
- `scan-vectors`: Number of different parameter combinations (default: 100)
- `points-per-vector`: Number of spatial points per scan vector (default: 1000)
- `config`: Path to configuration file with material properties

#### 1.2 Running FEA Simulations (if available)

If you have access to FEA solvers:

```bash
python src/fea_runner.py --config data/params.yaml --sweep parameter_sweep.csv
```

The `parameter_sweep.csv` file should contain parameter combinations to simulate:
```
P,v,h,theta
200,800,0.1,45
250,900,0.08,60
...
```

#### 1.3 Preprocessing Data

If you have raw simulation outputs, preprocess them:

```bash
python src/preprocessing.py --config data/params.yaml --torch
```

This converts raw FEA outputs to a structured HDF5 dataset ready for PINN training.

### 2. Train the PINN Surrogate Model

Train the physics-informed neural network:

```bash
python src/pinn/train.py --config data/params.yaml
```

The training process saves model checkpoints to `data/models/TIMESTAMP/checkpoints/`. The best performing model is saved as `best_model.pt`.

#### 2.1 Training with Physics Constraints

The physics-informed aspect of training is controlled in `params.yaml`:
```yaml
training:
  lambda_heat: 0.1   # Weight for heat equation physics loss
  lambda_stress: 0.1 # Weight for stress equation physics loss
```

Increase these weights to enforce stronger physics constraints, or decrease them to rely more on the data.

#### 2.2 Monitoring Training

Monitor training metrics with:
```bash
tensorboard --logdir data/models
```

To resume training from a checkpoint:
```bash
python src/pinn/train.py --config data/params.yaml --checkpoint path/to/checkpoint.pt
```

### 3. Run Multi-objective Optimization

Once you have a trained PINN surrogate model, run the optimizer:

```bash
python src/optimiser/nsga3.py --config data/params.yaml --model data/models/TIMESTAMP/checkpoints/best_model.pt
```

This generates a Pareto front of optimal process parameters in `data/optimized/pareto_solutions.h5` and `data/optimized/pareto_solutions.csv`.

#### 3.1 NSGA-III Optimization

The NSGA-III algorithm is configured in `params.yaml`:
```yaml
optimizer:
  pop_size: 100
  n_gen: 100
  n_partitions: 12
```

- `pop_size`: Population size (larger = more diverse solutions but slower)
- `n_gen`: Number of generations (more = better convergence but slower)
- `n_partitions`: Controls reference point density (affects solution diversity)

#### 3.2 Bayesian Optimization Alternative

Alternatively, use Bayesian optimization:
```bash
python src/optimiser/bayesopt.py --config data/params.yaml --model data/models/TIMESTAMP/checkpoints/best_model.pt
```

Bayesian optimization is typically more efficient for expensive evaluations but may not capture the full Pareto front as well as NSGA-III.

### 4. Validate Results (Optional)

If you have access to LPBF manufacturing equipment:

```bash
python src/validate/build_runner.py --config data/params.yaml --params data/optimized/pareto_solutions.csv --coupon cube --size 10 --dry-run
```

Remove the `--dry-run` flag to actually send the build job to the machine.

After building test coupons, characterize them:

```bash
python src/validate/characterise.py --config data/params.yaml --xct path/to/xct_data.tiff --ebsd path/to/ebsd_data.ang --prediction data/optimized/pareto_solutions.h5
```

## Project Structure

```
lpbf-optimizer/
├── data/
│   ├── raw/               <- FEA outputs, synchrotron frames
│   ├── processed/         <- tensors ready for NN
│   ├── models/            <- Trained model checkpoints
│   ├── optimized/         <- Optimization results
│   └── params.yaml        <- Configuration parameters
├── src/
│   ├── fea_runner.py      <- wraps ABAQUS / COMSOL jobs
│   ├── preprocessing.py   <- convert .odb/.vtk to torch tensors
│   ├── generate_synthetic_data.py <- Creates synthetic datasets
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
├── requirements.txt       <- Project dependencies
└── README.md
```

## Configuration

All configuration is managed through the `data/params.yaml` file, which includes:

- Material properties (thermal and mechanical)
- Neural network architecture
- Training hyperparameters
- FEA solver settings
- Optimization parameters
- Parameter bounds for optimization
- Validation settings

### Key Configuration Parameters

#### Material Properties
```yaml
material_properties:
  rho: 4430.0       # Density (kg/m^3)
  cp: 526.3         # Specific heat capacity (J/kg·K)
  k: 6.7            # Thermal conductivity (W/m·K)
  eta: 0.35         # Laser absorption coefficient
  r0: 0.05          # Laser beam radius (mm)
  # ... other properties
```

#### Neural Network Architecture
```yaml
model:
  input_dim: 10     # Process parameters + spatial coordinates + time
  output_dim: 3     # Residual stress, porosity, geometric accuracy
  hidden_width: 512 # Width of hidden layers
  hidden_depth: 5   # Number of hidden layers
```

#### Process Parameter Bounds
```yaml
optimizer:
  param_bounds:
    P: [150, 400]     # Laser power (W)
    v: [100, 2000]    # Scan speed (mm/s)
    h: [0.05, 0.15]   # Hatch spacing (mm)
    theta: [0, 90]    # Scan angle (degrees)
    # ... other parameters
```

You can create multiple configuration files for different materials or experimental setups.

## Detailed Module Descriptions

### 1. Data Generation and Processing

- **fea_runner.py**: Interface to commercial FEA solvers (ABAQUS, COMSOL)
  - Submits parametric studies to external solvers
  - Processes solver outputs into standardized formats
  - Supports parallel execution for high-throughput simulation

- **preprocessing.py**: Converts raw FEA outputs to structured data for ML
  - Extracts fields like temperature, stress, displacement
  - Performs spatial interpolation for consistent grid points
  - Normalizes data to suitable ranges for neural networks

- **generate_synthetic_data.py**: Creates synthetic datasets for testing/development
  - Implements simplified physics models for rapid data generation
  - Provides controllable noise levels for robustness testing
  - Creates realistic spatial correlations in outputs

### 2. Physics-Informed Neural Network

- **model.py**: Neural network architecture definition
  - Uses smooth activations (SiLU) for better gradient flow
  - Flexible architecture with configurable width and depth
  - Maps process parameters to material outcomes

- **physics.py**: Implements physical constraints (heat equation, stress equilibrium)
  - Uses automatic differentiation to compute PDE terms
  - Enforces physics constraints as additional loss terms
  - Handles boundary and initial conditions

- **train.py**: Training loop with combined data and physics losses
  - Balances empirical data fit with physics-based regularization
  - Implements learning rate scheduling and gradient clipping
  - Tracks and visualizes separate loss components

### 3. Optimization

- **nsga3.py**: Non-dominated Sorting Genetic Algorithm III for multi-objective optimization
  - Finds Pareto-optimal solutions for competing objectives
  - Uses reference points for diverse solution sets
  - Handles constraints and bounds on process parameters

- **bayesopt.py**: Bayesian optimization alternative for expensive evaluations
  - Uses Gaussian processes to model objective functions
  - Actively selects most informative points to evaluate
  - Efficiently handles noisy objective functions

### 4. Validation

- **build_runner.py**: Generates machine instructions for LPBF equipment
  - Converts optimized parameters to machine-specific G-code
  - Implements standard test coupon geometries
  - Supports various machine formats

- **characterise.py**: Analyzes experimental data from built parts
  - Processes XCT data for porosity measurement
  - Analyzes EBSD for residual stress and microstructure
  - Compares measured outcomes with model predictions

## Workflow Examples

### Basic Workflow Example

```bash
# Generate synthetic data for development
python src/generate_synthetic_data.py --config data/params.yaml

# Train the PINN model
python src/pinn/train.py --config data/params.yaml

# Run optimization to find optimal parameters
python src/optimiser/nsga3.py --config data/params.yaml --model data/models/*/checkpoints/best_model.pt
```

### Advanced Example: Transfer Learning to New Material

```bash
# Create a new configuration file for the new material
cp data/params.yaml data/params_in718.yaml

# Edit material properties in params_in718.yaml

# Fine-tune a pre-trained model
python src/pinn/train.py --config data/params_in718.yaml --checkpoint data/models/*/checkpoints/best_model.pt
```

### Example: Sensitivity Analysis Workflow

```bash
# Generate base synthetic dataset
python src/generate_synthetic_data.py --config data/params.yaml --scan-vectors 200

# Create notebook for sensitivity analysis
jupyter notebook notebooks/sensitivity_analysis.ipynb

# Train models with different physics weights
python src/pinn/train.py --config data/params_modified.yaml
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: 
   - Reduce batch size in params.yaml
   - Use gradient accumulation (increase `accumulation_steps`)
   - Train on CPU with `export CUDA_VISIBLE_DEVICES=''`

2. **FEA solver errors**: 
   - Check solver installation and template files
   - Verify solver license is active
   - Check for conflicting processes using the same license

3. **Unstable training**: 
   - Adjust learning rate (try 1e-4 instead of 1e-3)
   - Modify physics loss weights (start with lower values)
   - Use gradient clipping (set `clip_grad: true` and `clip_value: 1.0`)

4. **Poor optimization results**: 
   - Check parameter bounds for realistic values
   - Increase population size and generations
   - Examine surrogate model accuracy (should have R² > 0.8)

5. **Memory issues with large datasets**:
   - Use chunked data loading (set `use_data_loader: true`)
   - Reduce precision (use float16 with `mixed_precision: true`)
   - Process data in batches

### Logging

All modules use Python's logging system. To increase verbosity:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

You can also direct logs to a file:

```python
logging.basicConfig(filename='lpbf-optimizer.log', level=logging.INFO)
```

## Theory: Why Physics-Informed Beats Black-Box Models

* **Smaller data need** – PDE loss acts like an "infinite synthetic dataset".  
* **Better extrapolation** – network can't predict thermodynamically impossible states.  
* **Regulatory confidence** – engineers (and certifying bodies) can read the residual terms and trust the model.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## Core Technologies

- **Language**: Python 3.11
- **Libraries**: PyTorch, torchdiffeq, pymoo, numpy, scipy, matplotlib, h5py

## Implementation Timeline

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.