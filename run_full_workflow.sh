#!/bin/bash
# Run full KHI simulation workflow
# This script will:
# 1. Generate parameters for KHI simulations
# 2. Run a small test batch
# 3. Optionally run the full batch
# 4. Prepare data for diffusion model training

# Set the number of CPU cores to use for parallel processing
NUM_CORES=4  # Change this to match your CPU

# Step 1: Generate parameters for 1000 simulations
echo "========== STEP 1: Generating simulation parameters =========="
python generate_khi_parameters.py

# Step 2: Run a small test batch (first 5 simulations)
echo -e "\n========== STEP 2: Running test batch (5 simulations) =========="
python run_batch_simulations.py --start 0 --num 5 --processes 1

# Ask user if they want to continue with the full batch
echo -e "\n========== Test batch completed =========="
read -p "Do you want to run the full batch of simulations (995 remaining)? (y/n): " run_full_batch

if [[ $run_full_batch == "y" || $run_full_batch == "Y" ]]; then
    # Step 3: Run the remaining simulations in parallel
    echo -e "\n========== STEP 3: Running full batch (995 simulations) =========="
    echo "Using $NUM_CORES CPU cores for parallel processing"
    python run_batch_simulations.py --start 5 --num 995 --processes $NUM_CORES
else
    echo "Skipping full batch run"
fi

# Ask user if they want to prepare the data for diffusion model training
read -p "Do you want to prepare the data for diffusion model training? (y/n): " prepare_data

if [[ $prepare_data == "y" || $prepare_data == "Y" ]]; then
    # Step 4: Prepare data for diffusion model training
    echo -e "\n========== STEP 4: Preparing data for diffusion model training =========="
    
    # Ask which frames to use
    echo "Which frames do you want to use for training?"
    echo "1) Final frame only (default)"
    echo "2) All frames"
    echo "3) Selected frames"
    read -p "Enter your choice (1-3): " frame_choice
    
    if [[ $frame_choice == "2" ]]; then
        python prepare_diffusion_data.py --time_frames all
    elif [[ $frame_choice == "3" ]]; then
        read -p "Enter frame indices separated by commas (e.g., 0,50,100,150,200): " frames
        python prepare_diffusion_data.py --time_frames selected --selected_frames $frames
    else
        python prepare_diffusion_data.py --time_frames final
    fi
    
    # Ask if user wants to create TFRecords
    read -p "Do you want to create TFRecord files for TensorFlow? (requires TensorFlow) (y/n): " create_tfrecord
    
    if [[ $create_tfrecord == "y" || $create_tfrecord == "Y" ]]; then
        python prepare_diffusion_data.py --create_tfrecord
    fi
else
    echo "Skipping data preparation"
fi

echo -e "\n========== Workflow completed =========="
echo "You can find your simulation results in the khi_simulations directory"
echo "Prepared data for diffusion model training is in the diffusion_training_data directory" 