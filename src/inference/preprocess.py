#####################################################
# Inference Step 1: 
# - defines image preprocessing transform and 
# - hardcodes class information based on target model
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

def get_class_info():       
        
        # ---- Class information (HARDCODED as the labels of the target, as the surrogate is trained with the same soft labels) ----
        class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
        num_classes = len(class_names)

        return class_names, num_classes