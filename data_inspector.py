import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def inspect_npz_file(file_path):
    """Inspect the structure of a .npz file in detail"""
    data = np.load(file_path)

    print(f"\nExamining file: {os.path.basename(file_path)}")
    print(f"Keys in the file: {data.keys()}")

    # Print shape information
    for key in data.keys():
        print(f"Shape of '{key}': {data[key].shape}")

    # Sample array has shape (window_size, features, samples)
    sample_shape = data["sample"].shape
    label_shape = data["label"].shape

    print(f"\nWindow size: {sample_shape[0]}")
    print(f"Number of features: {sample_shape[1]}")
    print(f"Number of samples: {sample_shape[2]}")

    # Print some sample data
    print("\nFirst 5 RUL labels:", data["label"][:5])
    print("Last 5 RUL labels:", data["label"][-5:])

    # Distribution of RUL values
    plt.figure(figsize=(10, 5))
    plt.hist(data["label"], bins=50)
    plt.title(f"Distribution of RUL Values - {os.path.basename(file_path)}")
    plt.xlabel("RUL (Remaining Useful Life)")
    plt.ylabel("Frequency")
    plt.savefig(f"rul_dist_{os.path.basename(file_path).split('.')[0]}.png")
    plt.close()

    return data


def extract_cycles_from_original_h5(original_file, unit_number):
    """This would extract cycle information from the original H5 file

    Note: This is a placeholder - to use this function, you would need
    access to the original H5 file with the cycle information.
    """
    # This functionality would require access to the original H5 file
    print("This function requires access to the original N-CMAPSS_DS02-006.h5 file.")
    print("If you have this file, modify this function to extract cycle information.")

    # Mock data for demonstration
    return np.arange(1, 76) if unit_number == 2 else np.arange(1, 90)


def create_filtered_dataset(data, filter_function, output_file):
    """Create a new dataset filtered by a custom function

    Args:
        data: The original NPZ data
        filter_function: Function that takes a sample index and returns True/False
                        for inclusion in filtered dataset
        output_file: Path to save the filtered dataset
    """
    samples = data["sample"]
    labels = data["label"]

    # Determine which samples to keep
    indices_to_keep = [i for i in range(samples.shape[2]) if filter_function(i)]
    print(f"Keeping {len(indices_to_keep)} out of {samples.shape[2]} samples")

    # Create filtered arrays
    filtered_samples = samples[:, :, indices_to_keep]
    filtered_labels = labels[indices_to_keep]

    # Save to new file
    np.savez_compressed(output_file, sample=filtered_samples, label=filtered_labels)
    print(f"Saved filtered dataset to {output_file}")


def filter_by_range(min_idx, max_idx):
    """Create a filter function that selects samples by index range"""
    return lambda idx: min_idx <= idx < max_idx


def filter_by_rul_threshold(data, rul_threshold):
    """Create a filter function that selects samples with RUL <= threshold"""
    return lambda idx: data["label"][idx] <= rul_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and filter NPZ files from N-CMAPSS dataset"
    )
    parser.add_argument("file", type=str, help="Path to the NPZ file to inspect")
    parser.add_argument(
        "--filter-range", type=str, help="Filter by index range (start:end)"
    )
    parser.add_argument(
        "--filter-rul",
        type=float,
        help="Filter by RUL threshold (keep samples with RUL <= threshold)",
    )
    parser.add_argument(
        "--output-file", type=str, help="Output file for filtered dataset"
    )

    args = parser.parse_args()

    # Inspect the file
    data = inspect_npz_file(args.file)

    # Apply filtering if requested
    if args.filter_range or args.filter_rul:
        if not args.output_file:
            print("Error: --output-file is required when filtering")
            return

        if args.filter_range:
            try:
                start, end = map(int, args.filter_range.split(":"))
                filter_func = filter_by_range(start, end)
                create_filtered_dataset(data, filter_func, args.output_file)
            except ValueError:
                print("Error: --filter-range should be in format start:end")

        elif args.filter_rul:
            filter_func = filter_by_rul_threshold(data, args.filter_rul)
            create_filtered_dataset(data, filter_func, args.output_file)


if __name__ == "__main__":
    main()
