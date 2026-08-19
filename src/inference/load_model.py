#####################################################
# Inference Step 2: 
# - loads the model from saved checkpoint and 
# - changes to evaluation mode
#####################################################
from pathlib import Path
from src.model import (
    get_model,
    get_device
)
from src.train import load_model_checkpoint
import src.config as cfg

def load_model():
    device = get_device()

    # 1. load checkpoint first
    checkpoint = load_model_checkpoint(cfg.MODEL_CHECKPOINT_FILE, device)

    # 2. read class information from checkpoint
    class_names = checkpoint["class_names"] # class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")

    # 3. create model with correct number of outputs
    model = get_model(cfg.MODEL_NAME, num_classes)

    # 4. load trained weights
    model.load_state_dict(checkpoint["model"])

    # 5. move to device and evaluation mode
    model = model.to(device)
    model.eval() # for testing

    return model, class_names


