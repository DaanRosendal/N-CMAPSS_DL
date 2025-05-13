import os
import numpy as np
import argparse
import h5py
import matplotlib.pyplot as plt


def get_cycle_info(h5_file_path, unit_number):
    """Extract cycle information from the original H5 file for a specific unit"""
    with h5py.File(h5_file_path, "r") as f:
        # Get auxiliary data
        a_dev = np.array(f["A_dev"])
        a_test = np.array(f["A_test"])

        # Combine all data
        all_data = np.vstack((a_dev, a_test))

        # Filter by unit
        unit_data = all_data[all_data[:, 0] == unit_number]

        # Extract unique cycles
        unique_cycles = np.unique(unit_data[:, 1])

        return unique_cycles, unit_data


def visualize_cycles(unit_data, unit_number):
    """Create visualization of cycles and RUL"""
    plt.figure(figsize=(12, 6))

    # Group by cycle
    cycles = np.unique(unit_data[:, 1])
    cycle_counts = [np.sum(unit_data[:, 1] == c) for c in cycles]

    plt.bar(cycles, cycle_counts)
    plt.xlabel("Cycle Number")
    plt.ylabel("Data Points per Cycle")
    plt.title(f"Data Distribution Across Cycles for Unit {int(unit_number)}")
    plt.savefig(f"unit_{int(unit_number)}_cycles.png")
    plt.close()

    return cycles, cycle_counts


def map_indices_to_cycles(npz_file_path, h5_file_path, unit_number):
    """Map indices in NPZ file to cycles in original data"""
    # Load NPZ file
    npz_data = np.load(npz_file_path)
    samples = npz_data["sample"]
    labels = npz_data["label"]

    # Get cycle information from H5 file
    cycles, unit_data = get_cycle_info(h5_file_path, unit_number)

    # Create a mapping from RUL values to approximate cycles
    # This is an estimate since the exact mapping isn't stored in the NPZ files

    # The RUL values in labels represent time-to-failure in cycles
    # Max RUL should correspond to the first cycle, and 0 to the last cycle
    max_rul = np.max(labels)
    total_cycles = len(cycles)

    print(f"Unit {unit_number} has {total_cycles} cycles with max RUL {max_rul}")

    # Create a rough mapping
    # This isn't perfect but gives an approximation
    cycle_mapping = {}
    for i, label in enumerate(labels):
        # Estimate which cycle this sample belongs to
        # Higher RUL = earlier cycles
        cycle_idx = int((1 - (label / max_rul)) * (total_cycles - 1))
        estimated_cycle = cycles[cycle_idx]

        if estimated_cycle not in cycle_mapping:
            cycle_mapping[estimated_cycle] = []

        cycle_mapping[estimated_cycle].append(i)

    return cycle_mapping, cycles


def filter_by_max_cycle(
    npz_file_path, h5_file_path, unit_number, max_cycle, output_file
):
    """
    Filter data to exclude cycles above a certain number

    Args:
        npz_file_path: Path to the NPZ file
        h5_file_path: Path to the original H5 file
        unit_number: Unit number to filter
        max_cycle: Maximum cycle to include (exclude cycles > max_cycle)
        output_file: Path to save filtered data
    """
    # Map indices to cycles
    cycle_mapping, all_cycles = map_indices_to_cycles(
        npz_file_path, h5_file_path, unit_number
    )

    # Load NPZ data
    data = np.load(npz_file_path)
    samples = data["sample"]
    labels = data["label"]

    # Determine which cycles to include
    cycles_to_include = [c for c in all_cycles if c <= max_cycle]
    print(f"Including cycles: {cycles_to_include}")

    # Get indices of samples to keep
    indices_to_keep = []
    for cycle in cycles_to_include:
        if cycle in cycle_mapping:
            indices_to_keep.extend(cycle_mapping[cycle])

    print(f"Keeping {len(indices_to_keep)} out of {samples.shape[2]} samples")

    if not indices_to_keep:
        print("No samples to keep. Check your max_cycle value.")
        return

    # Create filtered arrays
    filtered_samples = samples[:, :, indices_to_keep]
    filtered_labels = labels[indices_to_keep]

    # Save to new file
    np.savez_compressed(output_file, sample=filtered_samples, label=filtered_labels)

    print(f"Saved filtered dataset to {output_file}")

    # Create visualization of RUL distribution before and after filtering
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=50, alpha=0.5, label="Original")
    plt.hist(filtered_labels, bins=50, alpha=0.5, label="Filtered")
    plt.xlabel("RUL")
    plt.ylabel("Frequency")
    plt.title(f"RUL Distribution Before and After Filtering (Max Cycle: {max_cycle})")
    plt.legend()
    plt.savefig(f"rul_distribution_unit{unit_number}_max_cycle{max_cycle}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Filter N-CMAPSS data by cycle number")
    parser.add_argument(
        "--npz-file",
        type=str,
        required=True,
        help="Path to the NPZ file (e.g., Unit2_win50_str1_smp10.npz)",
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        default="N-CMAPSS/N-CMAPSS_DS02-006.h5",
        help="Path to the original H5 file",
    )
    parser.add_argument("--unit", type=int, required=True, help="Unit number to filter")
    parser.add_argument(
        "--show-cycles",
        action="store_true",
        help="Show cycle information without filtering",
    )
    parser.add_argument(
        "--max-cycle",
        type=int,
        help="Maximum cycle to include (exclude cycles above this)",
    )
    parser.add_argument("--output-file", type=str, help="Path to save filtered data")

    args = parser.parse_args()

    if args.show_cycles:
        # Just show the cycle information
        cycles, unit_data = get_cycle_info(args.h5_file, args.unit)
        visualize_cycles(unit_data, args.unit)
        print(f"Unit {args.unit} has cycles: {cycles}")
    elif args.max_cycle is not None:
        if args.output_file is None:
            print("Error: --output-file is required when filtering")
            return

        # Filter by max cycle
        filter_by_max_cycle(
            args.npz_file, args.h5_file, args.unit, args.max_cycle, args.output_file
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
