####################################################################
# Step 0: defines the configurations.
####################################################################

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# settings for target API query
TARGET_API_URL = "http://127.0.0.1:8000/predict"
TARGET_QUERY_RESULTS_DIR = PROJECT_ROOT / "data" / "Query_Results"
TARGET_QUERY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_QUERY_RESULTS_FILE = TARGET_QUERY_RESULTS_DIR / "query_results.jsonl"

# settings for dataset split
TRAIN_DATASET_FILE = TARGET_QUERY_RESULTS_DIR / "train.jsonl"
VAL_DATASET_FILE = TARGET_QUERY_RESULTS_DIR / "val.jsonl"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# data preprocessing setting
DATA_DIR = "data/TrashBox"
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # ImageNet normalization (mandatory for pretrained models)
IMAGENET_STD  = [0.229, 0.224, 0.225]   # ImageNet normalization (mandatory for pretrained models)

# model settings for used for training, prediction
MODEL_NAME = "mobilenet_v2"         #lightweight model for training an extracted model
# MODEL_NAME = "resnet18"           # primary - main CNN result
# MODEL_NAME = "resnet34"           # baseline
# MODEL_NAME = "efficientnet_b0"    # best final model
OPTIMIZER_NAME = "adam"
# OPTIMIZER_NAME = "sgd"
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.01              # only with efficientnet_b0

# settings for model state saving
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINT_FILE = CHECKPOINTS_DIR / "mobilenet_v2_kldivLoss_adam_lr_0.001_epoch_10_batch_32_extracted_model_state.pth"
TRAINING_HISTORY_JSON = CHECKPOINTS_DIR / "mobilenet_v2_kldivLoss_adam_lr_0.001_epoch_10_batch_32_extracted_model_training_history.json"

