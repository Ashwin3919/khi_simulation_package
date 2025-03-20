# KHI Simulation Dataset Generator

This repository contains scripts to generate, run, and prepare Kelvin-Helmholtz Instability (KHI) simulations for diffusion model training. The system allows for generating thousands of different parameter combinations, running simulations efficiently, and organizing the results for machine learning.

## Overview

The system consists of several scripts:

1. `generate_khi_parameters.py` - Generates random parameter sets for KHI simulations
2. `run_khi_simulation.py` - Runs a single KHI simulation with specific parameters
3. `run_batch_simulations.py` - Runs multiple simulations in batch (supports parallel processing)
4. `prepare_diffusion_data.py` - Prepares the simulation data for diffusion model training
5. `khi_with_200_output.py` - Original KHI simulation script (reference only)

## Requirements

```
numpy
pandas
matplotlib
tqdm
```

For TFRecord generation (optional):
```
tensorflow
```

## Usage

### Step 1: Generate Simulation Parameters

Generate 1000 different parameter sets for KHI simulations:

```bash
python generate_khi_parameters.py
```

This creates a CSV file `khi_simulations/khi_parameters.csv` with 1000 different parameter combinations.

### Step 2: Run Simulations

#### Run a Single Simulation

To run a specific simulation by its ID:

```bash
python run_khi_simulation.py --id khi_sim_0001
```

Or by its row index in the CSV:

```bash
python run_khi_simulation.py --row 0
```

#### Run Multiple Simulations in Batch

To run multiple simulations (e.g., the first 10):

```bash
python run_batch_simulations.py --start 0 --num 10
```

To run simulations in parallel (e.g., using 4 processes):

```bash
python run_batch_simulations.py --start 0 --num 10 --processes 4
```

To randomly select 20 simulations:

```bash
python run_batch_simulations.py --num 20 --random
```

### Step 3: Prepare Data for Diffusion Model Training

After running simulations, prepare the data for diffusion model training:

```bash
python prepare_diffusion_data.py --sim_dir khi_simulations --output_dir diffusion_training_data
```

By default, only the final state of each simulation is used. Other options:

- To use all time frames:
  ```bash
  python prepare_diffusion_data.py --time_frames all
  ```

- To use specific frames:
  ```bash
  python prepare_diffusion_data.py --time_frames selected --selected_frames 0,50,100,150,200
  ```

- To create TFRecord files (requires TensorFlow):
  ```bash
  python prepare_diffusion_data.py --create_tfrecord
  ```

## Parameter Variations

The following parameters are varied:

| Parameter | Description | Range |
|-----------|-------------|-------|
| w0 | Perturbation amplitude | 0.05 - 0.3 |
| sigma | Width of perturbation | 0.01 - 0.1 |
| rho_contrast | Density contrast | 0.5 - 2.0 |
| vx_contrast | Velocity contrast | 0.2 - 1.0 |
| sinusoidal_freq | Frequency for sin function | 2 - 8 |
| P | Pressure | 1.0 - 4.0 |
| tEnd | End time | 1.0 - 3.0 |
| useSlopeLimiting | Boolean for slope limiting | 0 or 1 |

The parameter gamma (5/3) is kept constant across all simulations.

## Output Structure

Each simulation creates a directory structure like:

```
khi_simulations/
  ├── khi_sim_0001/
  │   ├── frame_0000.png
  │   ├── frame_0001.png
  │   ├── ...
  │   ├── density_0000.npy
  │   ├── density_0001.npy
  │   ├── ...
  │   ├── final_density.npy
  │   ├── parameters.csv
  │   ├── metadata.txt
  │   └── summary.txt
  ├── khi_sim_0002/
  │   ├── ...
  ...
```

The prepared data for diffusion model training is structured as:

```
diffusion_training_data/
  ├── train/
  │   ├── khi_sim_0001.npy
  │   ├── khi_sim_0001.png
  │   ├── khi_sim_0001.json
  │   ├── ...
  ├── val/
  │   ├── ...
  ├── test/
  │   ├── ...
  ├── dataset_metadata.csv
  └── dataset_info.json
```

## Example Workflow

```bash
# 1. Generate parameters for 1000 simulations
python generate_khi_parameters.py

# 2. Run first 10 simulations for testing
python run_batch_simulations.py --start 0 --num 10

# 3. After verifying, run more simulations in parallel
python run_batch_simulations.py --start 10 --num 990 --processes 8

# 4. Prepare data for diffusion model training
python prepare_diffusion_data.py --time_frames final
```

## Tips for Large-Scale Simulations

1. Start with a small batch to verify everything works correctly
2. Use the `--processes` flag for parallel processing (set to the number of CPU cores)
3. For very large runs, consider breaking them into multiple batches
4. Monitor disk space usage, as the full dataset can grow large if saving all frames
5. Use the `--random` flag to get a representative sample without running all simulations 