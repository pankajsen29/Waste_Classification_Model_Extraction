#####################################################
# Inference Step 1: 
# - defines image preprocessing transform and 
#####################################################

from torchvision import transforms

#ImageNet normalization (mandatory for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


#same as the Validation / Test transforms (NO augmentation, only resize + normalize)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])