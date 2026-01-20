from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk

# Path to your batches
batch_dir = Path("isl_synthetic_ocr_output_v2/_batches")

# Get all batch directories sorted
batch_paths = sorted(batch_dir.glob("batch_*"))
print(f"Found {len(batch_paths)} batches")

# Load all batches
print("Loading batches...")
all_datasets = []
for i, batch_path in enumerate(batch_paths):
    try:
        ds = load_from_disk(str(batch_path))
        all_datasets.append(ds)
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/{len(batch_paths)} batches")
    except Exception as e:
        print(f"  Error loading {batch_path}: {e}")

print(f"Successfully loaded {len(all_datasets)} batches")

# Concatenate all batches
print("Concatenating batches...")
final_dataset = concatenate_datasets(all_datasets)
print(f"Total examples: {len(final_dataset)}")

# Create train/test/validation split (80/10/10)
print("Creating splits...")
split_dataset = final_dataset.train_test_split(test_size=0.2, seed=42)
test_valid = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict(
    {
        "train": split_dataset["train"],
        "test": test_valid["test"],
        "validation": test_valid["train"],
    }
)

print(f"Train: {len(dataset_dict['train'])}")
print(f"Test: {len(dataset_dict['test'])}")
print(f"Validation: {len(dataset_dict['validation'])}")

# Save locally first (optional but recommended)
print("Saving to disk...")
dataset_dict.save_to_disk("isl_synthetic_ocr_output_v2/final")

# Push to Hugging Face Hub
print("Pushing to Hugging Face Hub...")
dataset_dict.push_to_hub("Sigurdur/isl_synthetic_ocr_v2")
print("Done!")
