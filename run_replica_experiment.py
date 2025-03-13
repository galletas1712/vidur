import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import argparse
import multiprocessing
import sys


class Redirect:
    """Context manager for redirecting stdout/stderr to a file."""

    def __init__(self, file_path):
        self.file = open(file_path, "w")
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self.file
        sys.stderr = self.file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.file.close()


def run_simulation(
    prefill_replicas,
    decode_replicas,
    hybrid_replicas,
    output_dir,
    length_generator_type,
    log_file,
):
    """Run a single simulation with the specified replica configuration."""
    total_replicas = prefill_replicas + decode_replicas + hybrid_replicas

    # Redirect stdout/stderr to log file during simulation
    with Redirect(log_file) as _:
        from vidur.config.config import (
            DistributionShiftRequestLengthGeneratorConfig,
            FixedRequestLengthGeneratorConfig,
            SyntheticRequestGeneratorConfig,
            UniformRequestLengthGeneratorConfig,
            ClusterConfig,
            MetricsConfig,
        )
        from vidur.simulator import SimulationConfig, Simulator

        request_length_generator_config = None
        if length_generator_type == "fixed":
            request_length_generator_config = FixedRequestLengthGeneratorConfig()
        elif length_generator_type == "uniform":
            request_length_generator_config = UniformRequestLengthGeneratorConfig()
        elif length_generator_type == "shift":
            request_length_generator_config = (
                DistributionShiftRequestLengthGeneratorConfig()
            )
        else:
            assert (
                False
            ), f"Invalid request length generator type: {length_generator_type}"

        # Create configuration
        simulation_config = SimulationConfig(
            request_generator_config=SyntheticRequestGeneratorConfig(
                length_generator_config=request_length_generator_config,
            ),
            cluster_config=ClusterConfig(
                num_prefill_replicas=prefill_replicas,
                num_decode_replicas=decode_replicas,
                num_hybrid_replicas=hybrid_replicas,
            ),
            metrics_config=MetricsConfig(output_dir=output_dir),
        )

        print(
            f"Running simulation with {total_replicas} replicas: "
            f"Prefill={prefill_replicas}, Decode={decode_replicas}, Hybrid={hybrid_replicas}"
        )

        simulator = Simulator(simulation_config)
        simulator.run()

    # Return the output directory which includes the timestamp
    return get_only_output_dir(output_dir)


def get_only_output_dir(base_output_dir):
    """Get the only directory within the base directory. Assert error if multiple directories exist."""
    dirs = [d for d in glob.glob(f"{base_output_dir}/*") if os.path.isdir(d)]
    if not dirs:
        return None
    assert (
        len(dirs) == 1
    ), f"Expected exactly one directory in {base_output_dir}, but found {len(dirs)}"
    return dirs[0]


def discover_metric_stages(output_dir, metric_name):
    """Discover all stages available for a given metric in the output directory."""
    plots_dir = f"{output_dir}/plots"
    if not os.path.exists(plots_dir):
        return []
    
    # Look for files matching the pattern stage*_metric.csv
    stage_files = glob.glob(f"{plots_dir}/*_{metric_name}.csv")
    stages = [os.path.basename(f).split("_")[0] for f in stage_files]
    return sorted(stages)

def extract_metric_cdf(output_dir, metric_name, stage=None):
    """Extract CDF data for a specific metric from CSV file."""
    # If stage is provided, look for stage-prefixed file, otherwise use the standard name
    file_prefix = f"{stage}_" if stage else ""
    csv_file = f"{output_dir}/plots/{file_prefix}{metric_name}.csv"

    if not os.path.exists(csv_file):
        print(f"Metrics file not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    if metric_name not in df.columns:
        print(f"{metric_name} column not found in {csv_file}")
        return None
    if "cdf" not in df.columns:
        print(f"CDF column not found in {csv_file}")
        return None
    return df


def calculate_median_metric(df, metric_name):
    """Calculate the median value of a metric from the CDF data."""
    # Find the row where CDF is closest to 0.5 (median)
    median_idx = (df["cdf"] - 0.5).abs().idxmin()
    return df.loc[median_idx, metric_name]


def plot_cdfs(
    cdf_data_dict, metric_name, output_dir, request_length_generator_type
):
    """Plot CDFs from multiple configurations for a specific metric."""
    if request_length_generator_type == "shift":
        # Use the first available configuration output directory to discover stages
        first_config_dir = None
        for label, output_dir_path in results:
            if output_dir_path:
                first_config_dir = output_dir_path
                break
        
        if not first_config_dir:
            print("No configuration directories found to discover stages")
            return
        
        # Discover all stages for this metric
        stages = discover_metric_stages(first_config_dir, metric_name)
        if not stages:
            print(f"No stages found for metric {metric_name}")
            return
        
        # Create a subplot for each stage
        fig, axes = plt.subplots(
            1, len(stages), figsize=(6*len(stages), 6), sharex=False, sharey=True
        )
        if len(stages) == 1:
            axes = [axes]  # Make sure axes is always a list
        
        for i, stage in enumerate(stages):
            ax = axes[i]
            
            # Extract stage-specific data for each configuration
            stage_data_dict = {}
            for label, output_dir_path in results:
                if output_dir_path:
                    df = extract_metric_cdf(output_dir_path, metric_name, stage)
                    if df is not None:
                        stage_data_dict[label] = df
            
            # Calculate medians for this stage
            medians = {}
            for label, df in stage_data_dict.items():
                if df is not None:
                    medians[label] = calculate_median_metric(df, metric_name)
            
            # Separate hybrid from non-hybrid configurations
            hybrid_labels = [label for label in stage_data_dict.keys() if "Hybrid" in label]
            non_hybrid_labels = [label for label in stage_data_dict.keys() if "Hybrid" not in label]
            
            # Sort non-hybrid configurations by median and take top 5
            sorted_non_hybrid = sorted(
                non_hybrid_labels, key=lambda label: medians.get(label, float("inf"))
            )
            top_5_non_hybrid = sorted_non_hybrid[:5]
            
            # Generate blue shades for non-hybrid configs
            blue_shades = plt.cm.Blues(np.linspace(0.5, 0.9, len(top_5_non_hybrid)))
            
            # Plot top 5 non-hybrid configurations in blue shades
            for j, label in enumerate(top_5_non_hybrid):
                df = stage_data_dict[label]
                if df is not None:
                    ax.plot(
                        df[metric_name],
                        df["cdf"],
                        label=f"{label} (median: {medians[label]:.3f}s)",
                        color=blue_shades[j],
                    )
            
            # Plot hybrid configuration in red
            for label in hybrid_labels:
                df = stage_data_dict[label]
                if df is not None:
                    ax.plot(
                        df[metric_name],
                        df["cdf"],
                        label=f"{label} (median: {medians[label]:.3f}s)",
                        color="red",
                        linewidth=2.5,
                    )
            
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_xlabel(f'{metric_name.replace("_", " ").title()} (s)')
            
            if i == 0:  # Only add y-label to the leftmost subplot
                ax.set_ylabel("CDF")
            
            ax.set_title(f'{stage}: {metric_name.replace("_", " ").title()}')
            ax.legend(fontsize="small")
        
        fig.suptitle(
            f'CDF of {metric_name.replace("_", " ").title()} by Stage for Different Replica Configurations',
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the super title
        plt.savefig(f"{output_dir}/{metric_name}_stages_cdf_comparison.png", dpi=300)
        plt.close()
    else:
        # Original single plot code for other generator types
        plt.figure(figsize=(10, 6))

        # Calculate median for each configuration
        medians = {}
        for label, df in cdf_data_dict.items():
            if df is not None:
                medians[label] = calculate_median_metric(df, metric_name)

        # Separate hybrid from non-hybrid configurations
        hybrid_labels = [label for label in cdf_data_dict.keys() if "Hybrid" in label]
        non_hybrid_labels = [
            label for label in cdf_data_dict.keys() if "Hybrid" not in label
        ]

        # Sort non-hybrid configurations by median and take top 5
        sorted_non_hybrid = sorted(
            non_hybrid_labels, key=lambda label: medians.get(label, float("inf"))
        )
        top_5_non_hybrid = sorted_non_hybrid[:5]

        # Generate blue shades for non-hybrid configs
        blue_shades = plt.cm.Blues(np.linspace(0.5, 0.9, len(top_5_non_hybrid)))

        # Plot top 5 non-hybrid configurations in blue shades
        for i, label in enumerate(top_5_non_hybrid):
            df = cdf_data_dict[label]
            if df is not None:
                plt.plot(
                    df[metric_name],
                    df["cdf"],
                    label=f"{label} (median: {medians[label]:.3f}s)",
                    color=blue_shades[i],
                )

        # Plot hybrid configuration in red
        for label in hybrid_labels:
            df = cdf_data_dict[label]
            if df is not None:
                plt.plot(
                    df[metric_name],
                    df["cdf"],
                    label=f"{label} (median: {medians[label]:.3f}s)",
                    color="red",
                    linewidth=2.5,
                )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xlabel(f'{metric_name.replace("_", " ").title()} (s)')
        plt.ylabel("CDF")
        plt.title(
            f'CDF of {metric_name.replace("_", " ").title()} for Different Replica Configurations'
        )
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{output_dir}/{metric_name}_cdf_comparison.png", dpi=300)
        plt.close()


def run_experiment_worker(config_tuple):
    """Worker function to run a single experiment configuration."""
    prefill, decode, hybrid, base_output_dir, request_length_generator_type = (
        config_tuple
    )

    # Create subdirectory for this configuration
    config_dir = f"{base_output_dir}/prefill_{prefill}_decode_{decode}_hybrid_{hybrid}"
    os.makedirs(config_dir, exist_ok=True)

    # Create log file path
    log_file = f"{config_dir}/simulation.log"

    # Run simulation
    output_dir = run_simulation(
        prefill, decode, hybrid, config_dir, request_length_generator_type, log_file
    )

    # Return configuration label and output directory for metric extraction
    if output_dir:
        if hybrid > 0:
            label = f"Hybrid only ({hybrid})"
        else:
            label = f"Prefill: {prefill}, Decode: {decode}"

        return label, output_dir

    return None, None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run replica experiments with configurable total GPUs."
    )
    parser.add_argument(
        "--total_replicas",
        type=int,
        default=4,
        help="Total number of replicas to use (default: 4)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )
    parser.add_argument(
        "--request_length_generator_type",
        type=str,
        default="shift",
        choices=["fixed", "uniform", "shift"],
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "request_e2e_time",
            "request_e2e_time_normalized",
            "prefill_e2e_time",
            "prefill_e2e_time_normalized",
            "decode_e2e_time",
            "decode_e2e_time_normalized",
            "tbt",
        ],
        help="Metrics to analyze and plot",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Create a base output directory for this experiment
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = f"simulator_output/replica_experiment_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Define configurations to test
    # Each tuple is (prefill_replicas, decode_replicas, hybrid_replicas)
    total_replicas = args.total_replicas
    request_length_generator_type = args.request_length_generator_type
    configs = [
        (
            num_prefill,
            total_replicas - num_prefill,
            0,
            base_output_dir,
            request_length_generator_type,
        )
        for num_prefill in range(1, total_replicas)
    ]
    configs.append(
        (0, 0, total_replicas, base_output_dir, request_length_generator_type)
    )

    # Initialize data structure for all metrics
    all_metrics_data = {metric: {} for metric in args.metrics}

    # Create a pool of workers
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # Map the worker function to the configurations
        global results
        results = pool.map(run_experiment_worker, configs)

        # Extract metrics for each configuration
        for label, output_dir in results:
            if label is not None and output_dir is not None:
                for metric in args.metrics:
                    if request_length_generator_type != "shift":
                        cdf = extract_metric_cdf(output_dir, metric)
                        if cdf is not None:
                            all_metrics_data[metric][label] = cdf

    # Plot comparison of CDFs for each metric
    for metric in args.metrics:
        if all_metrics_data[metric] or request_length_generator_type == "shift":
            plot_cdfs(
                all_metrics_data[metric],
                metric,
                base_output_dir,
                request_length_generator_type,
            )

    print(f"Experiment complete. Results saved in {base_output_dir}")


if __name__ == "__main__":
    main()
