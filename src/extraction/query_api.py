#####################################################
# Extraction Step 1: 
# - iterate over query images
# - call target API
# - save predictions in query_results.jsonl
# - output: query_results.jsonl
#
# - example json received:
# {
#   "image": "train/plastic/image_01.jpg",
#   "predicted_class": "Plastic",
#   "scores": 
#   {
#     "Plastic": 0.91,
#     "Paper": 0.04,
#     "Glass": 0.03,
#     "Metal": 0.02
#   }
# }
# 
# - example json stored:
# {
#   "image": "train/plastic/image_01.jpg",
#   "predicted_class": "Plastic",
#   "scores": 
#   {
#     "Plastic": 0.91,
#     "Paper": 0.04,
#     "Glass": 0.03,
#     "Metal": 0.02
#   }
#   "true_class": "plastic"
# }
#####################################################

import json
import requests
from pathlib import Path

# Configuration
API_URL = "http://127.0.0.1:8000/predict"
DATA_DIR = "data/TrashBox"
OUTPUT_FILE = "data/Query_Results/query_results.jsonl"

def get_query_results():
    """
    Query API and store results
    """
    query_status = True
    with open(Path(OUTPUT_FILE), "w", encoding="utf-8") as outfile:

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for image_path in Path(DATA_DIR).rglob("*"):

            if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
                continue
            
            relative_path = image_path.relative_to(DATA_DIR)
            #print(f"Processing: {relative_path}")
            true_class = relative_path.parts[1]

            mime_types = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png"
                    }
            
            # "image_path" should be used for filesystem access, not the "relative_path"
            content_type = mime_types.get(image_path.suffix.lower(), "application/octet-stream")

            try:
                with open(image_path, "rb") as img_file:

                    files = {
                        "file": (
                            str(relative_path.as_posix()),
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
                prediction_result["true_class"] = true_class
                outfile.write(json.dumps(prediction_result) + "\n")

            except Exception as e:
                query_status = False
                #print(f"Failed: {relative_path} -> {e}")
    
    return OUTPUT_FILE, query_status
    