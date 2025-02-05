import random
import json
import datasets
import datasets.config

# Parameters
sample_size = 20_000
dataset_name = "wikimedia/wikipedia"
config_name = "20231101.en"

datasets.config.TORCH_AVAILABLE = False

# Load the dataset in streaming mode
dataset = datasets.load_dataset(dataset_name, config_name, streaming=True)

# Reservoir sampling: as we iterate over the stream, we maintain a sample of fixed size.
reservoir = []
num_processed = 0

print("Starting to sample pages from the dataset...")
for example in dataset["train"]:  # The dataset has a 'train' split.
    num_processed += 1

    if len(reservoir) < sample_size:
        # Initially fill the reservoir.
        reservoir.append(example)
    else:
        # Once reservoir is full, replace an element with decreasing probability.
        j = random.randint(0, num_processed - 1)
        if j < sample_size:
            reservoir[j] = example

    # Optional: print progress every 100,000 pages processed.
    if num_processed % 100_000 == 0:
        print(f"Processed {num_processed} pages...")

print(f"Finished processing. Total pages processed: {num_processed}")
print(f"Collected a random sample of {len(reservoir)} pages.")

# Save the sample to a JSON Lines file.
output_filename = "wikipedia_20k_sample.jsonl"
with open(output_filename, "w", encoding="utf-8") as outfile:
    for entry in reservoir:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Sample saved to {output_filename}")
