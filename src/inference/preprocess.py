#####################################################
# Inference Step 1: 
# - defines image preprocessing transform and 
#####################################################

from torchvision import transforms
import src.config as cfg

#same as the Validation / Test transforms (NO augmentation, only resize + normalize)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
])