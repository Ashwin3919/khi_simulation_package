import numpy as np
import pandas as pd
import os

# Number of different parameter sets to generate
num_simulations = 1000

# Define parameter ranges (min, max) for each parameter
param_ranges = {
    'w0': (0.05, 0.3),           # Perturbation amplitude
    'sigma': (0.01, 0.1),        # Width of perturbation
    'rho_contrast': (0.5, 2.0),  # Density contrast (added to base density)
    'vx_contrast': (0.2, 1.0),   # Velocity contrast
    'sinusoidal_freq': (2, 8),   # Frequency for sin function (instead of fixed 4*pi)
    'P': (1.0, 4.0),             # Pressure
    'tEnd': (1.0, 3.0),          # End time
    'useSlopeLimiting': (0, 1)   # Boolean for slope limiting (0 or 1)
}

# Fixed parameter
gamma = 5/3.

# Generate random parameters within specified ranges
np.random.seed(42)  # For reproducibility

# Initialize dictionary to store parameters
parameters = {key: [] for key in param_ranges.keys()}
parameters['simulation_id'] = []
parameters['gamma'] = []

for i in range(num_simulations):
    # Generate unique simulation ID
    sim_id = f"khi_sim_{i+1:04d}"
    parameters['simulation_id'].append(sim_id)
    
    # Generate random values for each parameter
    for param, (min_val, max_val) in param_ranges.items():
        if param == 'useSlopeLimiting':
            # This is a boolean parameter
            val = np.random.choice([0, 1])
        elif param == 'sinusoidal_freq':
            # This should be an integer
            val = np.random.randint(min_val, max_val+1)
        else:
            # Continuous parameter
            val = np.random.uniform(min_val, max_val)
        parameters[param].append(val)
    
    # Add fixed gamma
    parameters['gamma'].append(gamma)

# Create a DataFrame and save to CSV
df = pd.DataFrame(parameters)

# Create output directory if it doesn't exist
if not os.path.exists('khi_simulations'):
    os.makedirs('khi_simulations')

# Save to CSV
csv_path = 'khi_simulations/khi_parameters.csv'
df.to_csv(csv_path, index=False)

print(f"Generated {num_simulations} parameter sets saved to {csv_path}")
print("\nSample of the first 5 parameter sets:")
print(df.head(5)) 