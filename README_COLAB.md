# KHI Simulation Dataset Generator for Google Colab

This repository contains scripts to generate, run, and prepare Kelvin-Helmholtz Instability (KHI) simulations for diffusion model training, adapted to work with Google Colab and Google Drive.

## Google Colab Setup

1. Create a new Google Colab notebook
2. Mount Google Drive by running:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Clone the repository (if using git) or upload the required files:
```python
# Option 1: Clone the repository
!git clone <your-repository-url>

# Option 2: Upload files manually
# Upload these files to your Colab workspace:
# - KHI_Simulation_Colab.py
# - generate_khi_parameters.py
# - run_batch_simulations.py
# - run_khi_simulation.py
# - prepare_diffusion_data.py
```

4. Navigate to the working directory:
```python
%cd /content/khi_simulation_package  # or your uploaded directory
```

## Running Simulations in Colab

Here's a complete example notebook setup:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies (if not already installed)
!pip install numpy pandas matplotlib tqdm tensorflow

# 3. Run first batch (simulations 0-9)
!python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 0

# 4. Run second batch (simulations 10-19)
!python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 10

# 5. Run third batch (simulations 20-29)
!python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 20
```

### Tips for Colab Usage

1. **Save the Notebook**: Create a copy of the notebook in your Google Drive:
   - Click `File > Save a copy in Drive`
   - This allows you to reuse the notebook for future runs

2. **Monitor Resources**:
   - Click `Runtime > Resource usage` to monitor RAM and GPU usage
   - If resources are low, restart the runtime: `Runtime > Restart runtime`

3. **Handle Long Runs**:
   - Keep your browser tab active
   - Consider using a browser extension to prevent Colab from disconnecting
   - Save intermediate results between batches

4. **Check Progress**:
   - Monitor the output in your Google Drive
   - Each batch will create its own set of simulation directories
   - Check the logs in real-time

5. **Using GPU Acceleration**:
   - Go to `Runtime > Change runtime type`
   - Select "GPU" as Hardware accelerator
   - This can speed up certain computations

### Example Colab Workflow

```python
# Setup cell
from google.colab import drive
drive.mount('/content/drive')
!pip install numpy pandas matplotlib tqdm tensorflow

# First batch - morning run
!python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 0 --processes 2

# Check results in Drive
!ls -l /content/drive/MyDrive/khi_simulations/

# Second batch - afternoon run
!python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 10 --processes 2

# Third batch - evening run
!python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 20 --processes 2
```

### Monitoring Progress

You can check the progress of your simulations in Colab using:

```python
# List current simulation directories
!ls -l /content/drive/MyDrive/khi_simulations/

# Check the latest simulation logs
!tail -f /content/drive/MyDrive/khi_simulations/khi_sim_*/summary.txt

# View current simulation parameters
!cat /content/drive/MyDrive/khi_simulations/khi_parameters.csv
```

## Quick Start

1. Upload `KHI_Simulation_Colab.py` and all related scripts to your Google Colab environment
2. Run the script with desired parameters:
```bash
# Run first batch of 10 simulations
python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 0
```
3. All outputs will be automatically saved to your Google Drive at `/content/drive/MyDrive/khi_simulations/`

## Command Line Arguments

The main script `KHI_Simulation_Colab.py` accepts the following arguments:

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| --processes | Number of parallel processes for simulations | 1 | Any positive integer |
| --time_frames | Which time frames to use for diffusion training | 'final' | 'final', 'all' |
| --num_simulations | Number of simulations to run in this batch | 10 | Any positive integer |
| --start_idx | Starting index for the simulation batch | 0 | Any non-negative integer |

## Example Usage

1. Run first batch of 10 simulations:
```bash
python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 0
```

2. Run second batch with multiple processes:
```bash
python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 10 --processes 4
```

3. Run third batch with all time frames:
```bash
python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 20 --processes 4 --time_frames all
```

## Sequential Batch Processing

To run multiple batches of simulations sequentially, you can run the script multiple times with increasing start indices. For example:

1. First batch (simulations 0-9):
```bash
python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 0
```

2. Second batch (simulations 10-19):
```bash
python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 10
```

3. Third batch (simulations 20-29):
```bash
python KHI_Simulation_Colab.py --num_simulations 10 --start_idx 20
```

This approach allows you to:
- Monitor progress between batches
- Manage Colab resources effectively
- Resume from where you left off if disconnected
- Verify results between batches

## Workflow Steps

The script automatically performs the following steps:

1. Mounts your Google Drive
2. Sets up the directory structure at `/content/drive/MyDrive/khi_simulations/`
3. Installs required dependencies:
   - numpy
   - pandas
   - matplotlib
   - tqdm
   - tensorflow
4. Generates 1000 parameter sets with varying conditions
5. Runs the simulations with specified parallelization
6. Prepares the data for diffusion model training
7. Saves all results to your Google Drive

## Output Structure

The simulation results will be saved to your Google Drive with this structure:

```
/content/drive/MyDrive/khi_simulations/
  ├── khi_parameters.csv           # Generated simulation parameters
  ├── khi_sim_0001/              # Individual simulation directories
  │   ├── frame_0000.png
  │   ├── ...
  │   ├── density_0000.npy
  │   ├── ...
  │   ├── final_density.npy
  │   ├── parameters.csv
  │   ├── metadata.txt
  │   └── summary.txt
  ├── khi_sim_0002/
  │   ├── ...
  ...

/content/drive/MyDrive/khi_simulations/diffusion_training_data/
  ├── train/                    # Training data split
  │   ├── khi_sim_0001.npy
  │   ├── khi_sim_0001.png
  │   ├── khi_sim_0001.json
  │   ├── ...
  ├── val/                      # Validation data split
  │   ├── ...
  ├── test/                     # Test data split
  │   ├── ...
  ├── dataset_metadata.csv      # Complete dataset metadata
  └── dataset_info.json         # Dataset configuration info
```

## Parameter Variations

The following parameters are automatically varied in the simulations:

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

## Tips for Using Colab

1. **Runtime Limitations**: Google Colab has runtime limitations (usually disconnects after 12 hours). For large simulation sets:
   - Use `--processes` to speed up computation
   - Run smaller batches if needed
   - Save progress frequently (automatic in this implementation)

2. **Resource Management**:
   - Monitor your RAM usage when running parallel processes
   - Reduce the number of processes if you encounter memory issues
   - Consider using Colab Pro for better resources

3. **Storage Considerations**:
   - Check your Google Drive storage before running large batches
   - Use `--time_frames final` to save only final states if storage is limited
   - Clean up intermediate files if not needed

## Troubleshooting

1. If the script fails to mount Google Drive:
   - Ensure you're running in Google Colab
   - Follow the authentication prompts when they appear
   - Try remounting manually

2. If simulations fail:
   - Check the logs in the simulation directories
   - Reduce the number of parallel processes
   - Ensure sufficient Colab runtime resources

3. For memory issues:
   - Reduce the number of parallel processes
   - Use `--time_frames final` instead of `all`
   - Clear Colab runtime memory between runs

## Contributing

Feel free to submit issues and enhancement requests! 