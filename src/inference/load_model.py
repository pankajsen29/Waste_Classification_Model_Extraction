#####################################################
# Inference Step 2: 
# - loads the model from saved checkpoint and 
# - changes to evaluation mode
#####################################################
from src.inference.preprocess import get_class_info
from src.model import (
    get_model,
    get_device
)
from src.train import load_trained_model

def load_model():
    class_names, num_classes = get_class_info()
    device = get_device()
    model = get_model("mobilenet_v2", num_classes) #lightweight model for training an extracted model
    #model = get_model("resnet18", num_classes) #primary - main CNN result
    #model = get_model("resnet34", num_classes) #baseline
    #model = get_model("efficientnet_b0", num_classes) #best final model
    model = model.to(device)


    from pathlib import Path
    checkpoint_dir = Path("checkpoints")

    model, optimizer = load_trained_model(model, checkpoint_dir / "extracted_waste_seg_full_training_state.pth")

    # for testing
    model.eval()

    return model


