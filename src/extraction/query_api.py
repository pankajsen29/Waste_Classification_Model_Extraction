#####################################################
# Extraction Step 1: 
# - iterate over query images
# - call target API
# - save predictions in query_results.jsonl
# - output: query_results.jsonl
#
# - example json received:
# {
#   "image": "iamge_01.jpg",
#   "predicted_class": "Plastic",
#   "scores": 
#   {
#     "Plastic": 0.91,
#     "Paper": 0.04,
#     "Glass": 0.03,
#     "Metal": 0.02
#   }
# }
#####################################################

import os
import json
import requests
from pathlib import Path

# Configuration
API_URL = "http://127.0.0.1:8000/predict"
DATA_DIR = "data/TrashBox"
OUTPUT_FILE = "query_results.jsonl"

# Query API and store results
with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for image_path in Path(DATA_DIR).rglob("*"):

        if image_path.suffix.lower() not in image_extensions:
            continue

        print(f"Processing: {image_path.name}")

        mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png"
                }
        
        content_type = mime_types.get(image_path.suffix.lower(), "application/octet-stream")

        try:
            with open(image_path, "rb") as img_file:

                files = {
                    "file": (
                        image_path.name,
                        img_file,
                        content_type
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=30
                )

            response.raise_for_status()

            prediction_result = response.json()

            outfile.write(json.dumps(prediction_result) + "\n")

        except Exception as e:
            print(f"Failed: {image_path.name} -> {e}")

print(f"\nFinished. Results stored in {OUTPUT_FILE}")