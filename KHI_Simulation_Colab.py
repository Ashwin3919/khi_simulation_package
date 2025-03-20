import os
import argparse
from google.colab import drive

def main(args):
    # Mount Google Drive
    drive.mount('/content/drive')

    # Set up directory structure
    base_dir = '/content/drive/MyDrive/khi_simulations/'
    os.makedirs(base_dir, exist_ok=True)

    # Install dependencies
    os.system('pip install numpy pandas matplotlib tqdm tensorflow')

    # Generate parameter sets with specified number of simulations
    os.system(f'python generate_khi_parameters.py --num_simulations {args.num_simulations}')

    # Run simulations with specified range
    os.system(f'python run_batch_simulations.py --processes {args.processes} --csv {os.path.join(base_dir, "khi_parameters.csv")} '
              f'--start {args.start_idx} --num {args.num_simulations}')

    # Prepare output for diffusion model training
    os.system(f'python prepare_diffusion_data.py --simulation_dir {base_dir} '
              f'--output_dir {os.path.join(base_dir, "diffusion_training_data")} '
              f'--time_frames {args.time_frames}')

    # Notify completion
    print(f'KHI Simulation workflow completed for simulations {args.start_idx} to {args.start_idx + args.num_simulations - 1}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KHI simulation workflow')
    parser.add_argument('--processes', type=int, default=1, help='Number of parallel processes for simulations')
    parser.add_argument('--time_frames', type=str, default='final', choices=['final', 'all'], 
                       help='Which time frames to use for diffusion training')
    parser.add_argument('--num_simulations', type=int, default=10, 
                       help='Number of simulations to run in this batch')
    parser.add_argument('--start_idx', type=int, default=0, 
                       help='Starting index for the simulation batch')
    
    args = parser.parse_args()
    main(args)