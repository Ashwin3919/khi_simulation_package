import os
import sys
import argparse
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import time
from datetime import datetime
import json
from run_khi_simulation import run_khi_simulation

def run_simulation_wrapper(sim_id=None, row_idx=None, csv_file=None):
    """Wrapper function for running a single simulation, used for multiprocessing"""
    try:
        output_dir = run_khi_simulation(sim_id=sim_id, param_row=row_idx, csv_file=csv_file)
        return {
            'sim_id': sim_id,
            'status': 'success',
            'output_dir': output_dir
        }
    except Exception as e:
        error_msg = str(e)
        print(f"Error in simulation {sim_id if sim_id else row_idx}: {error_msg}")
        return {
            'sim_id': sim_id if sim_id else f"row_{row_idx}",
            'status': 'failed',
            'error': error_msg
        }

def run_batch(start_idx=0, 
              end_idx=None, 
              num_simulations=None,
              csv_file='khi_simulations/khi_parameters.csv', 
              num_processes=1,
              random_selection=False):
    """
    Run a batch of KHI simulations.
    
    Args:
        start_idx: Starting index in the CSV file (default: 0)
        end_idx: Ending index in the CSV file (exclusive)
        num_simulations: Number of simulations to run (alternative to end_idx)
        csv_file: Path to the CSV file with parameters
        num_processes: Number of parallel processes to use
        random_selection: If True, randomly select simulations instead of sequential
    """
    if not os.path.exists(csv_file):
        print(f"Parameter file {csv_file} not found. Run generate_khi_parameters.py first.")
        return
    
    # Create the main directory if it doesn't exist
    main_dir = os.path.dirname(csv_file)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    
    # Load parameters from CSV
    df = pd.read_csv(csv_file)
    total_simulations = len(df)
    
    # Determine range of simulations to run
    if end_idx is None and num_simulations is None:
        end_idx = total_simulations
    elif end_idx is None:
        end_idx = min(start_idx + num_simulations, total_simulations)
    
    if end_idx > total_simulations:
        end_idx = total_simulations
        print(f"Warning: End index adjusted to {end_idx} (total available simulations)")
    
    # Create a batch directory to store results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(main_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Determine which simulations to run
    if random_selection:
        print(f"Randomly selecting {end_idx - start_idx} simulations")
        selected_indices = np.random.choice(total_simulations, end_idx - start_idx, replace=False)
        selected_indices.sort()  # Sort for better logging
    else:
        selected_indices = range(start_idx, end_idx)
    
    # Prepare simulation arguments
    sim_args = []
    for idx in selected_indices:
        sim_id = df.iloc[idx]['simulation_id']
        sim_args.append((sim_id, None, csv_file))
    
    # Create a batch info file
    batch_info = {
        'batch_id': f"batch_{timestamp}",
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'csv_file': csv_file,
        'total_in_csv': total_simulations,
        'num_selected': len(sim_args),
        'random_selection': random_selection,
        'simulation_ids': [arg[0] for arg in sim_args],
        'num_processes': num_processes,
    }
    
    with open(os.path.join(batch_dir, 'batch_info.json'), 'w') as f:
        json.dump(batch_info, f, indent=2)
    
    # Save the subset of parameters for this batch
    batch_df = df[df['simulation_id'].isin([arg[0] for arg in sim_args])]
    batch_df.to_csv(os.path.join(batch_dir, 'batch_parameters.csv'), index=False)
    
    # Run simulations
    results = []
    start_time = time.time()
    
    print(f"Starting batch run of {len(sim_args)} simulations with {num_processes} processes")
    
    if num_processes > 1:
        # Run in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in tqdm(pool.starmap(run_simulation_wrapper, sim_args), total=len(sim_args)):
                results.append(result)
    else:
        # Run sequentially
        for sim_id, row_idx, csv_path in tqdm(sim_args, total=len(sim_args)):
            result = run_simulation_wrapper(sim_id=sim_id, csv_file=csv_path)
            results.append(result)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Save results summary
    summary = {
        'batch_id': batch_info['batch_id'],
        'total_simulations': len(sim_args),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'elapsed_time': elapsed_time,
        'average_time_per_simulation': elapsed_time / len(sim_args) if sim_args else 0,
        'results': results
    }
    
    with open(os.path.join(batch_dir, 'batch_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\nBatch Simulation Summary:")
    print(f"Total simulations: {len(sim_args)}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per simulation: {summary['average_time_per_simulation']:.2f} seconds")
    print(f"Results saved to: {batch_dir}")
    
    return batch_dir, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run batch KHI simulations')
    parser.add_argument('--start', type=int, default=0, help='Starting index in the CSV file')
    parser.add_argument('--end', type=int, help='Ending index in the CSV file (exclusive)')
    parser.add_argument('--num', type=int, help='Number of simulations to run (alternative to --end)')
    parser.add_argument('--csv', type=str, default='khi_simulations/khi_parameters.csv', 
                        help='CSV file with simulation parameters')
    parser.add_argument('--processes', type=int, default=1, 
                        help='Number of parallel processes to use')
    parser.add_argument('--random', action='store_true', 
                        help='Randomly select simulations instead of sequential')
    
    args = parser.parse_args()
    
    try:
        batch_dir, results = run_batch(
            start_idx=args.start,
            end_idx=args.end,
            num_simulations=args.num,
            csv_file=args.csv,
            num_processes=args.processes,
            random_selection=args.random
        )
        print(f"Batch simulation completed. Results saved to {batch_dir}")
    except Exception as e:
        print(f"Error in batch run: {e}")
        sys.exit(1) 