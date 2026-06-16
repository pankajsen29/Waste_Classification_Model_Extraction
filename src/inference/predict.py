#####################################################
# Inference Step 3: 
# - receives PIL image
# - preprocesses
# - performs inference
# - computes softmax probabilities
# - returns ALL confidence scores
#
# - example json return:
# {
#   "predicted_class": "Plastic",
#   "scores": 
#   {
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
#####################################################

import torch
from PIL import Image

from src.model import get_device
from src.inference.preprocess import inference_transforms, get_class_info
from src.inference.load_model import load_model

# Load ONCE globally
model = load_model()

device = get_device()

class_names, num_classes = get_class_info()

def predict_image(image: Image.Image):
    # Preprocess image
    image_tensor = inference_transforms(image)

    # Add batch dimension for pytorch (e.g., shape: (1, 3, 224, 224) means: (batch_size=1, channels=3, height=224, width=224))
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Disable unnecessary gradient tracking for inference
    with torch.no_grad():

        outputs = model(image_tensor)

        # convert logits to probabilities 
        # hint: softmax is used because class: multiclass, loss function: CrossEntropyLoss [hint: model -> logits -> CrossEntropyLoss]
        probabilities = torch.softmax(outputs, dim=1)

        probabilities = probabilities[0]

    # convert tensor to python list [hint: python lists can only be created from CPU tensors safely]
    probs_list = probabilities.cpu().tolist()

    # find highest probability class: gets predicted class index
    predicted_index = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_index]

    # confidence scores dictionary
    scores = {}
    for class_name, score in zip(class_names, probs_list):
        scores[class_name] = round(score, 4)

    return {
        "predicted_class": predicted_class,
        "scores": scores
    }


