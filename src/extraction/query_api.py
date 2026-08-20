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
#    {
#      "Cardboard": 0.9753, 
#      "Food Organics": 0.0003, 
#      "Glass": 0.0003, 
#      "Metal": 0.0015, 
#      "Miscellaneous Trash": 0.0021, 
#      "Paper": 0.0153, 
#      "Plastic": 0.0004, 
#      "Textile Trash": 0.003, 
#      "Vegetation": 0.0018
#   }
# }
# 
# - example json stored:
# {
#   "image": "train/plastic/image_01.jpg",
#   "predicted_class": "Plastic",
#   "scores": 
#    {
#      "Cardboard": 0.9753, 
#      "Food Organics": 0.0003, 
#      "Glass": 0.0003, 
#      "Metal": 0.0015, 
#      "Miscellaneous Trash": 0.0021, 
#      "Paper": 0.0153, 
#      "Plastic": 0.0004, 
#      "Textile Trash": 0.003, 
#      "Vegetation": 0.0018
#   }
#   "true_class": "plastic"
# }
#####################################################

import json
import requests
from pathlib import Path
import src.config as cfg

def get_query_results():
    """
    Query API and store results
    """
    query_status = True
    with open(cfg.TARGET_QUERY_RESULTS_FILE, "w", encoding="utf-8") as outfile:

        for image_path in Path(cfg.DATA_DIR).rglob("*"):

            if not image_path.is_file() or image_path.suffix.lower() not in cfg.image_extensions:
                continue
            
            relative_path = image_path.relative_to(cfg.DATA_DIR)
            #print(f"Processing: {relative_path}")
            true_class = relative_path.parts[1]
            
            # "image_path" should be used for filesystem access, not the "relative_path"
            content_type = cfg.mime_types.get(image_path.suffix.lower(), "application/octet-stream")

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
                        cfg.TARGET_API_URL,
                        files=files,
                        timeout=30
                    )

                response.raise_for_status()
                prediction_result = response.json()
                prediction_result["true_class"] = true_class
                outfile.write(json.dumps(prediction_result) + "\n")
                query_status = True

            except Exception as e:
                query_status = False
                print(f"Failed: {relative_path} -> {e}")
    
    return cfg.TARGET_QUERY_RESULTS_FILE, query_status
    