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
import src.config as cfg

def get_dataset_splitted():
    """
    read all JSONL records from query_results.jsonl and split into train.jsonl (80% train) + val.jsonl (20% validation)
    """
    # Load records
    records = []

    with open(cfg.TARGET_QUERY_RESULTS_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    #print(f"Loaded {len(records)} records")

    # Shuffle
    random.seed(cfg.RANDOM_SEED)
    random.shuffle(records)

    # Split
    train_size = int(len(records) * cfg.TRAIN_RATIO)

    train_records = records[:train_size]
    val_records = records[train_size:]

    #print(f"Train samples: {len(train_records)}")
    #print(f"Validation samples: {len(val_records)}")

    # Save train.jsonl
    with open(cfg.TRAIN_DATASET_FILE, "w", encoding="utf-8") as outfile:
        for record in train_records:
            outfile.write(json.dumps(record) + "\n")

    # Save val.jsonl
    with open(cfg.VAL_DATASET_FILE, "w", encoding="utf-8") as outfile:
        for record in val_records:
            outfile.write(json.dumps(record) + "\n")

    return cfg.TRAIN_DATASET_FILE, cfg.VAL_DATASET_FILE