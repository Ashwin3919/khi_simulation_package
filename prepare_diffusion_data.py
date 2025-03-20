import os
import argparse
import numpy as np
import pandas as pd
import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

def prepare_dataset(simulation_dir, 
                    output_dir='diffusion_training_data',
                    split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1},
                    time_frames='final',  # 'final', 'all', 'selected', or int
                    selected_frames=None,  # List of frame indices if time_frames='selected'
                    normalize=True):
    """
    Prepare simulation data for diffusion model training.
    
    Args:
        simulation_dir: Directory containing the simulation results
        output_dir: Directory to save the prepared data
        split_ratios: Dictionary with ratios for train/val/test splits
        time_frames: Which frames to use: 'final' (only final frame), 'all' (all frames),
                     'selected' (specific frames), or int (specific frame number)
        selected_frames: List of frame indices if time_frames='selected'
        normalize: Whether to normalize the density fields
    """
    # Check if simulation directory exists
    if not os.path.exists(simulation_dir):
        raise FileNotFoundError(f"Simulation directory {simulation_dir} not found")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create split directories
    for split in split_ratios.keys():
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Get all simulation directories
    sim_dirs = []
    for item in os.listdir(simulation_dir):
        item_path = os.path.join(simulation_dir, item)
        if os.path.isdir(item_path) and item.startswith('khi_sim_'):
            sim_dirs.append(item_path)
    
    if not sim_dirs:
        raise ValueError(f"No simulation directories found in {simulation_dir}")
    
    print(f"Found {len(sim_dirs)} simulation directories")
    
    # Shuffle the directories to get a random split
    np.random.seed(42)
    np.random.shuffle(sim_dirs)
    
    # Split the directories according to the ratios
    total = len(sim_dirs)
    train_count = int(total * split_ratios['train'])
    val_count = int(total * split_ratios['val'])
    test_count = total - train_count - val_count
    
    splits = {
        'train': sim_dirs[:train_count],
        'val': sim_dirs[train_count:train_count+val_count],
        'test': sim_dirs[train_count+val_count:]
    }
    
    print(f"Split counts: Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Process each split
    all_metadata = []
    for split, directories in splits.items():
        print(f"Processing {split} split...")
        for sim_dir in tqdm(directories):
            sim_id = os.path.basename(sim_dir)
            
            # Load parameters from the CSV file
            params_file = os.path.join(sim_dir, 'parameters.csv')
            if not os.path.exists(params_file):
                print(f"Warning: Parameters file not found for {sim_id}, skipping")
                continue
            
            params = pd.read_csv(params_file).iloc[0].to_dict()
            
            # Determine which frames to process
            if time_frames == 'final':
                # Use the final density file
                density_file = os.path.join(sim_dir, 'final_density.npy')
                if not os.path.exists(density_file):
                    print(f"Warning: Final density file not found for {sim_id}, trying to find the last frame")
                    # Find the highest numbered frame
                    density_files = [f for f in os.listdir(sim_dir) if f.startswith('density_') and f.endswith('.npy')]
                    if not density_files:
                        print(f"Warning: No density files found for {sim_id}, skipping")
                        continue
                    density_file = os.path.join(sim_dir, sorted(density_files)[-1])
                
                # Process this single file
                process_density_file(density_file, sim_id, params, split, output_dir, normalize)
                all_metadata.append(params)
                
            elif time_frames == 'all':
                # Use all available density files
                density_files = [f for f in os.listdir(sim_dir) if f.startswith('density_') and f.endswith('.npy')]
                for density_file in sorted(density_files):
                    process_density_file(os.path.join(sim_dir, density_file), 
                                       f"{sim_id}_{os.path.splitext(density_file)[0]}", 
                                       params, split, output_dir, normalize)
                    all_metadata.append(params)
            
            elif time_frames == 'selected' and selected_frames is not None:
                # Use only the selected frames
                for frame_idx in selected_frames:
                    density_file = os.path.join(sim_dir, f'density_{frame_idx:04d}.npy')
                    if os.path.exists(density_file):
                        process_density_file(density_file, f"{sim_id}_frame_{frame_idx:04d}", 
                                           params, split, output_dir, normalize)
                        all_metadata.append(params)
                    else:
                        print(f"Warning: Frame {frame_idx} not found for {sim_id}")
            
            elif isinstance(time_frames, int):
                # Use a specific frame
                density_file = os.path.join(sim_dir, f'density_{time_frames:04d}.npy')
                if os.path.exists(density_file):
                    process_density_file(density_file, f"{sim_id}_frame_{time_frames:04d}", 
                                       params, split, output_dir, normalize)
                    all_metadata.append(params)
                else:
                    print(f"Warning: Frame {time_frames} not found for {sim_id}")
            
            else:
                print(f"Warning: Invalid time_frames option: {time_frames}")
    
    # Save metadata for all processed simulations
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'dataset_metadata.csv'), index=False)
    
    # Create dataset info file
    dataset_info = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_simulations': len(sim_dirs),
        'train_count': len(splits['train']),
        'val_count': len(splits['val']),
        'test_count': len(splits['test']),
        'time_frames': time_frames,
        'normalization': normalize,
        'split_ratios': split_ratios
    }
    
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset preparation complete!")
    print(f"Dataset saved to: {output_dir}")
    print(f"Total processed images: {len(all_metadata)}")

def process_density_file(density_file, sim_id, params, split, output_dir, normalize):
    """Process a single density file, save as image and numpy array"""
    # Load density data
    density = np.load(density_file)
    
    # Normalize if needed
    if normalize:
        density_min = density.min()
        density_max = density.max()
        density_norm = (density - density_min) / (density_max - density_min)
    else:
        density_norm = density
    
    # Save as NPY file for the model (raw data)
    np_output_path = os.path.join(output_dir, split, f"{sim_id}.npy")
    np.save(np_output_path, density)
    
    # Save as PNG for visualization
    plt.figure(figsize=(5, 5))
    plt.imshow(density_norm.T)
    plt.axis('off')
    plt.tight_layout()
    
    png_output_path = os.path.join(output_dir, split, f"{sim_id}.png")
    plt.savefig(png_output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Also save parameters as a JSON file
    json_output_path = os.path.join(output_dir, split, f"{sim_id}.json")
    with open(json_output_path, 'w') as f:
        json.dump(params, f, indent=2)

def create_tfrecord_dataset(input_dir, output_file):
    """
    Create a TFRecord dataset from the prepared data.
    This is useful for TensorFlow-based diffusion models.
    
    Requires TensorFlow to be installed.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed. Cannot create TFRecord dataset.")
        return
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(input_dir, split)
        output_path = f"{output_file}_{split}.tfrecord"
        
        with tf.io.TFRecordWriter(output_path) as writer:
            npy_files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
            
            for npy_file in tqdm(npy_files, desc=f"Creating {split} TFRecord"):
                # Load the numpy data
                npy_path = os.path.join(split_dir, npy_file)
                data = np.load(npy_path)
                
                # Load parameters if available
                json_path = os.path.join(split_dir, npy_file.replace('.npy', '.json'))
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        params = json.load(f)
                else:
                    params = {}
                
                # Convert to float32
                data = data.astype(np.float32)
                
                # Create the Example
                feature = {
                    'density': tf.train.Feature(
                        float_list=tf.train.FloatList(value=data.flatten())
                    ),
                    'shape': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=data.shape)
                    ),
                    'sim_id': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[npy_file.encode()])
                    )
                }
                
                # Add parameters as features
                for key, value in params.items():
                    if isinstance(value, (int, np.int64, np.int32)):
                        feature[key] = tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[value])
                        )
                    elif isinstance(value, (float, np.float64, np.float32)):
                        feature[key] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=[value])
                        )
                    elif isinstance(value, str):
                        feature[key] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[value.encode()])
                        )
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        
        print(f"Created TFRecord file: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare simulation data for diffusion model training')
    parser.add_argument('--sim_dir', type=str, default='khi_simulations',
                        help='Directory containing simulation results')
    parser.add_argument('--output_dir', type=str, default='diffusion_training_data',
                        help='Directory to save the prepared data')
    parser.add_argument('--time_frames', type=str, default='final',
                        help="Which frames to use: 'final', 'all', 'selected', or a frame number")
    parser.add_argument('--selected_frames', type=str, default=None,
                        help='Comma-separated list of frame indices if time_frames="selected"')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalize the density fields')
    parser.add_argument('--create_tfrecord', action='store_true',
                        help='Create a TFRecord dataset (requires TensorFlow)')
    
    args = parser.parse_args()
    
    # Convert selected_frames to a list of integers if provided
    if args.selected_frames:
        args.selected_frames = [int(idx) for idx in args.selected_frames.split(',')]
    
    # Convert time_frames to an integer if it's a number
    if args.time_frames.isdigit():
        args.time_frames = int(args.time_frames)
    
    try:
        prepare_dataset(
            simulation_dir=args.sim_dir,
            output_dir=args.output_dir,
            time_frames=args.time_frames,
            selected_frames=args.selected_frames,
            normalize=args.normalize
        )
        
        if args.create_tfrecord:
            create_tfrecord_dataset(args.output_dir, f"{args.output_dir}/khi_dataset")
            
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        import traceback
        traceback.print_exc() 