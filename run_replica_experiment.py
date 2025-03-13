import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import argparse
import multiprocessing

def run_simulation(prefill_replicas, decode_replicas, hybrid_replicas, output_dir, log_file):
    """Run a single simulation with the specified replica configuration."""
    # Base command with common parameters
    cmd = [
        "python", "-m", "vidur.main",
        f"--cluster_config_num_prefill_replicas={prefill_replicas}",
        f"--cluster_config_num_decode_replicas={decode_replicas}",
        f"--cluster_config_num_hybrid_replicas={hybrid_replicas}",
        f"--metrics_config_output_dir={output_dir}"
    ]
    
    # Open log file for redirecting output
    with open(log_file, 'w') as log:
        total_replicas = prefill_replicas + decode_replicas + hybrid_replicas
        log.write(f"Running simulation with {total_replicas} replicas: "
                f"Prefill={prefill_replicas}, Decode={decode_replicas}, Hybrid={hybrid_replicas}")
    
        # Run the simulation with redirected output
        try:
            subprocess.run(cmd, check=True, stdout=log, stderr=log)
        except Exception as e:
            log.write(f"Error occurred: {str(e)}\n")
    
    # Return the output directory which includes the timestamp
    return get_only_output_dir(output_dir)

def get_only_output_dir(base_output_dir):
    """Get the only directory within the base directory. Assert error if multiple directories exist."""
    dirs = [d for d in glob.glob(f"{base_output_dir}/*") if os.path.isdir(d)]
    if not dirs:
        return None
    assert len(dirs) == 1, f"Expected exactly one directory in {base_output_dir}, but found {len(dirs)}"
    return dirs[0]

def extract_request_e2e_time_cdf(output_dir):
    """Extract the request e2e time CDF data from CSV file."""
    csv_file = f"{output_dir}/plots/request_e2e_time.csv"
    if not os.path.exists(csv_file):
        print(f"Request metrics file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    assert 'request_e2e_time' in df.columns, f"request_e2e_time column not found in {csv_file}"
    assert 'cdf' in df.columns, f"CDF column not found in {csv_file}"
    return df

def calculate_median_e2e_time(df):
    """Calculate the median request_e2e_time from the CDF data."""
    # Find the row where CDF is closest to 0.5 (median)
    median_idx = (df['cdf'] - 0.5).abs().idxmin()
    return df.loc[median_idx, 'request_e2e_time']

def plot_cdfs(cdf_data_dict, output_dir):
    """Plot CDFs from multiple configurations."""
    plt.figure(figsize=(10, 6))
    
    # Calculate median for each configuration
    medians = {}
    for label, df in cdf_data_dict.items():
        if df is not None:
            medians[label] = calculate_median_e2e_time(df)
    
    # Separate hybrid from non-hybrid configurations
    hybrid_labels = [label for label in cdf_data_dict.keys() if "Hybrid" in label]
    non_hybrid_labels = [label for label in cdf_data_dict.keys() if "Hybrid" not in label]
    
    # Sort non-hybrid configurations by median and take top 5
    sorted_non_hybrid = sorted(non_hybrid_labels, key=lambda label: medians.get(label, float('inf')))
    top_5_non_hybrid = sorted_non_hybrid[:5]
    
    # Generate blue shades for non-hybrid configs
    blue_shades = plt.cm.Blues(np.linspace(0.5, 0.9, len(top_5_non_hybrid)))
    
    # Plot top 5 non-hybrid configurations in blue shades
    for i, label in enumerate(top_5_non_hybrid):
        df = cdf_data_dict[label]
        if df is not None:
            plt.plot(df['request_e2e_time'], df['cdf'], 
                     label=f"{label} (median: {medians[label]:.3f}s)", 
                     color=blue_shades[i])
    
    # Plot hybrid configuration in red
    for label in hybrid_labels:
        df = cdf_data_dict[label]
        if df is not None:
            plt.plot(df['request_e2e_time'], df['cdf'], 
                     label=f"{label} (median: {medians[label]:.3f}s)", 
                     color='red', linewidth=2.5)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Request End-to-End Time (s)')
    plt.ylabel('CDF')
    plt.title('CDF of Request End-to-End Time for Different Replica Configurations')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/request_e2e_time_cdf_comparison.png", dpi=300)
    plt.close()

def run_experiment_worker(config_tuple):
    """Worker function to run a single experiment configuration."""
    prefill, decode, hybrid, base_output_dir = config_tuple
    
    # Create subdirectory for this configuration
    config_dir = f"{base_output_dir}/prefill_{prefill}_decode_{decode}_hybrid_{hybrid}"
    os.makedirs(config_dir, exist_ok=True)
    
    # Create log file path
    log_file = f"{config_dir}/simulation.log"
    
    # Run simulation
    output_dir = run_simulation(prefill, decode, hybrid, config_dir, log_file)
    print(config_tuple, output_dir)
    
    # Extract CDF data
    if output_dir:
        cdf = extract_request_e2e_time_cdf(output_dir)
        if cdf is not None:
            if hybrid > 0:
                label = f"Hybrid only ({hybrid})"
            else:
                label = f"Prefill: {prefill}, Decode: {decode}"
            return label, cdf
    
    return None, None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run replica experiments with configurable total GPUs.')
    parser.add_argument('--total_replicas', type=int, default=8,
                        help='Total number of replicas to use (default: 8)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create a base output directory for this experiment
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_output_dir = f"simulator_output/replica_experiment_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Define configurations to test
    # Each tuple is (prefill_replicas, decode_replicas, hybrid_replicas)
    total_replicas = args.total_replicas
    configs = [
        (num_prefill, total_replicas - num_prefill, 0, base_output_dir)
        for num_prefill in range(1, total_replicas)
    ]
    configs.append((0, 0, total_replicas, base_output_dir))
    
    # Run simulations in parallel and collect CDF data
    cdf_data = {}
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # Map the worker function to the configurations
        results = pool.map(run_experiment_worker, configs)
        
        # Collect the results
        for label, cdf in results:
            print("LAbel", label)
            print(cdf)
            if label is not None and cdf is not None:
                cdf_data[label] = cdf
    
    # Plot comparison of CDFs
    print(cdf_data)
    plot_cdfs(cdf_data, base_output_dir)
    print(f"Experiment complete. Results saved in {base_output_dir}")
    
if __name__ == "__main__":
    main()
