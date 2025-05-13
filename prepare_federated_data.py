import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
import h5py
from sklearn.preprocessing import StandardScaler


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], np.integer):
            return [int(x) for x in obj]
        return super().default(obj)


def load_unit_data(data_dir, unit_number):
    """Load data for a specific unit"""
    file_path = os.path.join(data_dir, f"Unit{unit_number}_win50_str1_smp10.npz")
    data = np.load(file_path)
    return data


def get_cycle_mapping(npz_file_path, h5_file_path, unit_number):
    """Map indices in NPZ file to cycles in original data"""
    # Load NPZ file
    npz_data = np.load(npz_file_path)
    labels = npz_data["label"]

    # Get cycle information from H5 file
    with h5py.File(h5_file_path, "r") as f:
        # Get auxiliary data
        a_dev = np.array(f["A_dev"])
        a_test = np.array(f["A_test"])

        # Combine all data
        all_data = np.vstack((a_dev, a_test))

        # Filter by unit
        unit_data = all_data[all_data[:, 0] == unit_number]

        # Extract unique cycles
        cycles = np.unique(unit_data[:, 1])

    # The RUL values in labels represent time-to-failure in cycles
    # Max RUL should correspond to the first cycle, and 0 to the last cycle
    max_rul = np.max(labels)
    total_cycles = len(cycles)

    # Create a rough mapping
    cycle_mapping = {}
    for i, label in enumerate(labels):
        # Estimate which cycle this sample belongs to
        # Higher RUL = earlier cycles
        cycle_idx = int((1 - (label / max_rul)) * (total_cycles - 1))
        if cycle_idx >= len(cycles):
            cycle_idx = len(cycles) - 1
        estimated_cycle = cycles[cycle_idx]

        if estimated_cycle not in cycle_mapping:
            cycle_mapping[estimated_cycle] = []

        cycle_mapping[estimated_cycle].append(i)

    return cycle_mapping, cycles


def filter_by_max_cycle(data, unit_number, max_cycle, h5_file_path):
    """Filter data to exclude cycles above a certain number"""
    npz_file_path = f"N-CMAPSS/Samples_whole/Unit{unit_number}_win50_str1_smp10.npz"

    # Get cycle mapping
    cycle_mapping, all_cycles = get_cycle_mapping(
        npz_file_path, h5_file_path, unit_number
    )

    # Determine cycles to include
    cycles_to_include = [c for c in all_cycles if c <= max_cycle]

    # Get samples to keep
    samples = data["sample"]
    labels = data["label"]

    indices_to_keep = []
    for cycle in cycles_to_include:
        if cycle in cycle_mapping:
            indices_to_keep.extend(cycle_mapping[cycle])

    if not indices_to_keep:
        print(
            f"Warning: No samples to keep for unit {unit_number} with max_cycle {max_cycle}"
        )
        return None, None

    # Filter data
    filtered_samples = samples[:, :, indices_to_keep]
    filtered_labels = labels[indices_to_keep]

    return filtered_samples, filtered_labels


def prepare_client_data(
    data_dir,
    unit_numbers,
    output_dir,
    n_clients,
    split_method="by_unit",
    rul_threshold=None,
    max_cycle_dict=None,
    h5_file_path=None,
):
    """
    Prepare data for federated learning experiments

    Args:
        data_dir: Directory containing the NPZ files
        unit_numbers: List of unit numbers to include
        output_dir: Directory to save client data
        n_clients: Number of clients to create
        split_method: How to split the data:
            - 'by_unit': Each unit becomes a separate client
            - 'by_rul': Split each unit by RUL ranges
            - 'random': Random assignment to clients
        rul_threshold: Optional RUL threshold to filter data
        max_cycle_dict: Dictionary mapping unit numbers to maximum cycle to include
        h5_file_path: Path to original H5 file (required if max_cycle_dict is provided)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Track client statistics for visualization
    client_stats = {}

    if split_method == "by_unit":
        # Each unit is a separate client
        for i, unit in enumerate(unit_numbers):
            client_id = i % n_clients
            client_dir = os.path.join(output_dir, f"client_{client_id}")
            os.makedirs(client_dir, exist_ok=True)

            # Load unit data
            data = load_unit_data(data_dir, unit)
            samples = data["sample"]
            labels = data["label"]

            # Filter by max cycle if specified
            if max_cycle_dict and unit in max_cycle_dict and h5_file_path:
                max_cycle = max_cycle_dict[unit]
                print(f"Filtering unit {unit} to include cycles <= {max_cycle}")
                filtered_samples, filtered_labels = filter_by_max_cycle(
                    data, unit, max_cycle, h5_file_path
                )

                if filtered_samples is not None:
                    samples = filtered_samples
                    labels = filtered_labels

            # Filter by RUL if specified
            if rul_threshold is not None:
                indices = labels <= rul_threshold
                samples = samples[:, :, indices]
                labels = labels[indices]

            # Save to client directory
            output_file = os.path.join(client_dir, f"unit_{unit}_data.npz")
            np.savez_compressed(output_file, sample=samples, label=labels)

            # Track statistics
            if client_id not in client_stats:
                client_stats[client_id] = {
                    "units": [],
                    "n_samples": 0,
                    "rul_min": float("inf"),
                    "rul_max": 0,
                }

            client_stats[client_id]["units"].append(unit)
            client_stats[client_id]["n_samples"] += len(labels)
            client_stats[client_id]["rul_min"] = min(
                client_stats[client_id]["rul_min"], np.min(labels)
            )
            client_stats[client_id]["rul_max"] = max(
                client_stats[client_id]["rul_max"], np.max(labels)
            )

    elif split_method == "by_rul":
        # Split each unit's data by RUL ranges
        rul_bins = np.linspace(0, 100, n_clients + 1)

        for unit in unit_numbers:
            # Load unit data
            data = load_unit_data(data_dir, unit)
            samples = data["sample"]
            labels = data["label"]

            # Filter by max cycle if specified
            if max_cycle_dict and unit in max_cycle_dict and h5_file_path:
                max_cycle = max_cycle_dict[unit]
                print(f"Filtering unit {unit} to include cycles <= {max_cycle}")
                filtered_samples, filtered_labels = filter_by_max_cycle(
                    data, unit, max_cycle, h5_file_path
                )

                if filtered_samples is not None:
                    samples = filtered_samples
                    labels = filtered_labels

            # Split by RUL ranges
            for i in range(n_clients):
                min_rul = rul_bins[i]
                max_rul = rul_bins[i + 1]

                # Find samples in this RUL range
                indices = (labels >= min_rul) & (labels < max_rul)

                if np.sum(indices) > 0:  # Only proceed if we have samples
                    client_dir = os.path.join(output_dir, f"client_{i}")
                    os.makedirs(client_dir, exist_ok=True)

                    # Extract and save data for this client
                    client_samples = samples[:, :, indices]
                    client_labels = labels[indices]

                    output_file = os.path.join(
                        client_dir, f"unit_{unit}_rul_{int(min_rul)}_{int(max_rul)}.npz"
                    )
                    np.savez_compressed(
                        output_file, sample=client_samples, label=client_labels
                    )

                    # Track statistics
                    if i not in client_stats:
                        client_stats[i] = {
                            "units": [],
                            "n_samples": 0,
                            "rul_min": float("inf"),
                            "rul_max": 0,
                        }

                    if unit not in client_stats[i]["units"]:
                        client_stats[i]["units"].append(unit)

                    client_stats[i]["n_samples"] += len(client_labels)
                    client_stats[i]["rul_min"] = min(
                        client_stats[i]["rul_min"], np.min(client_labels)
                    )
                    client_stats[i]["rul_max"] = max(
                        client_stats[i]["rul_max"], np.max(client_labels)
                    )

    elif split_method == "random":
        # Randomly assign data to clients
        all_samples = None  # Use None instead of [] for empty check
        all_labels = None
        all_units = None

        # First, load and concatenate all data
        for unit in unit_numbers:
            data = load_unit_data(data_dir, unit)
            samples = data["sample"]
            labels = data["label"]

            # Filter by max cycle if specified
            if max_cycle_dict and unit in max_cycle_dict and h5_file_path:
                max_cycle = max_cycle_dict[unit]
                print(f"Filtering unit {unit} to include cycles <= {max_cycle}")
                filtered_samples, filtered_labels = filter_by_max_cycle(
                    data, unit, max_cycle, h5_file_path
                )

                if filtered_samples is not None:
                    samples = filtered_samples
                    labels = filtered_labels

            # Filter by RUL if specified
            if rul_threshold is not None:
                indices = labels <= rul_threshold
                samples = samples[:, :, indices]
                labels = labels[indices]

            # Track unit for each sample for statistical purposes
            unit_indices = np.full(len(labels), unit)

            # Add to collection
            if all_samples is None:
                all_samples = samples
                all_labels = labels
                all_units = unit_indices
            else:
                all_samples = np.concatenate((all_samples, samples), axis=2)
                all_labels = np.concatenate((all_labels, labels))
                all_units = np.concatenate((all_units, unit_indices))

        # Shuffle indices
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(all_labels))
        samples_per_client = len(indices) // n_clients

        for i in range(n_clients):
            client_dir = os.path.join(output_dir, f"client_{i}")
            os.makedirs(client_dir, exist_ok=True)

            # Determine indices for this client
            start_idx = i * samples_per_client
            end_idx = (
                min((i + 1) * samples_per_client, len(indices))
                if i < n_clients - 1
                else len(indices)
            )
            client_indices = indices[start_idx:end_idx]

            # Extract data
            client_samples = all_samples[:, :, client_indices]
            client_labels = all_labels[client_indices]
            client_units = all_units[client_indices]

            # Save to client directory
            output_file = os.path.join(client_dir, f"random_data.npz")
            np.savez_compressed(output_file, sample=client_samples, label=client_labels)

            # Track statistics
            unique_units = np.unique(client_units)
            client_stats[i] = {
                "units": list(unique_units),
                "n_samples": len(client_labels),
                "rul_min": np.min(client_labels),
                "rul_max": np.max(client_labels),
            }

    # Save client statistics
    with open(os.path.join(output_dir, "client_stats.json"), "w") as f:
        json.dump(client_stats, f, indent=4, cls=NumpyEncoder)

    # Create visualization of client data distribution
    visualize_client_distribution(client_stats, output_dir, split_method)

    return client_stats


def visualize_client_distribution(client_stats, output_dir, split_method):
    """Create visualizations of the client data distribution"""

    # Plot number of samples per client
    plt.figure(figsize=(10, 6))
    client_ids = sorted(client_stats.keys())
    samples = [client_stats[cid]["n_samples"] for cid in client_ids]
    plt.bar(client_ids, samples)
    plt.title(f"Number of Samples per Client ({split_method})")
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.savefig(os.path.join(output_dir, f"samples_per_client_{split_method}.png"))
    plt.close()

    # Plot RUL range per client
    plt.figure(figsize=(10, 6))
    rul_min = [client_stats[cid]["rul_min"] for cid in client_ids]
    rul_max = [client_stats[cid]["rul_max"] for cid in client_ids]

    plt.bar(client_ids, np.array(rul_max) - np.array(rul_min), bottom=rul_min)
    plt.title(f"RUL Range per Client ({split_method})")
    plt.xlabel("Client ID")
    plt.ylabel("RUL Range")
    plt.savefig(os.path.join(output_dir, f"rul_range_per_client_{split_method}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare data for federated learning")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="N-CMAPSS/Samples_whole",
        help="Directory containing the NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="federated_data",
        help="Directory to save client data",
    )
    parser.add_argument(
        "--n-clients", type=int, default=6, help="Number of clients to create"
    )
    parser.add_argument(
        "--split-method",
        type=str,
        choices=["by_unit", "by_rul", "random"],
        default="by_unit",
        help="Method to split data among clients",
    )
    parser.add_argument(
        "--train-units",
        type=str,
        default="2,5,10,16,18,20",
        help="Comma-separated list of training unit numbers",
    )
    parser.add_argument(
        "--rul-threshold", type=float, help="Optional RUL threshold to filter data"
    )
    parser.add_argument(
        "--max-cycles",
        type=str,
        help="Maximum cycle per unit, format: unit:cycle,unit:cycle (e.g., 2:50,5:60)",
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        default="N-CMAPSS/N-CMAPSS_DS02-006.h5",
        help="Path to the original H5 file (required if using --max-cycles)",
    )

    args = parser.parse_args()

    # Parse unit numbers
    train_units = [int(x) for x in args.train_units.split(",")]

    # Parse max cycle dictionary if provided
    max_cycle_dict = None
    if args.max_cycles:
        max_cycle_dict = {}
        for item in args.max_cycles.split(","):
            unit, cycle = map(int, item.split(":"))
            max_cycle_dict[unit] = cycle

    # Prepare client data
    client_stats = prepare_client_data(
        args.data_dir,
        train_units,
        args.output_dir,
        args.n_clients,
        args.split_method,
        args.rul_threshold,
        max_cycle_dict,
        args.h5_file,
    )

    print(
        f"Data prepared for {len(client_stats)} clients using {args.split_method} split method"
    )
    print(f"Client data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
