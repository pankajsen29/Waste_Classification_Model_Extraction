#######################################################################################
# Extraction Step 2: 
# - reads all JSONL records from query_results.jsonl
# - and shuffle them.
# - splits into train.jsonl (80% train) + val.jsonl (20% validation)
# - output: train.jsonl and val.jsonl
#######################################################################################

import json
import random
from pathlib import Path

# Configuration
INPUT_FILE = "data/Query_Results/query_results.jsonl"

TRAIN_FILE = "data/Query_Results/train.jsonl"
VAL_FILE = "data/Query_Results/val.jsonl"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Load records
records = []

with open(Path(INPUT_FILE), "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip()

        if not line:
            continue

        records.append(json.loads(line))

print(f"Loaded {len(records)} records")


# Shuffle
random.seed(RANDOM_SEED)
random.shuffle(records)


# Split
train_size = int(len(records) * TRAIN_RATIO)

train_records = records[:train_size]
val_records = records[train_size:]

print(f"Train samples: {len(train_records)}")
print(f"Validation samples: {len(val_records)}")


# Save train.jsonl
with open(Path(TRAIN_FILE), "w", encoding="utf-8") as outfile:
    for record in train_records:
        outfile.write(json.dumps(record) + "\n")


# Save val.jsonl
with open(Path(VAL_FILE), "w", encoding="utf-8") as outfile:
    for record in val_records:
        outfile.write(json.dumps(record) + "\n")

print("\nDataset split is completed")
print(f"Train file: {Path(TRAIN_FILE).resolve()}")
print(f"Validation file: {Path(VAL_FILE).resolve()}")