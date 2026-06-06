#######################################################################################
# Extraction Step 3: 
# - converts json into a dataset structure which PyTorch Dataset class can easily load.
#######################################################################################
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Configuration
DATA_DIR = "data/TrashBox"
TRAIN_FILE = "data/Query_Results/train.jsonl"
VAL_FILE = "data/Query_Results/val.jsonl"

#ImageNet normalization (mandatory for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


#training transforms (with augmentation)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

#Validation transforms (NO augmentation, only resize + normalize)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

class JSONLDataset(Dataset):
    """
    Reads image paths and labels from a .jsonl file.
    Each line: {"image": "plastic/img_001.jpg", "predicted_class": "plastic", "scores": {...}}
    Images are resolved relative to DATA_DIR.
    """
    def __init__(self, jsonl_path: str, data_dir: str, class_to_idx: dict, transform=None):
        self.data_dir     = data_dir
        self.class_to_idx = class_to_idx
        self.transform    = transform
        self.records      = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    # returns image: tensor([3,224,224]), target_vector: tensor([0.05,0.04,0.91])
    def __getitem__(self, idx):
        record   = self.records[idx]
        img_path = os.path.join(self.data_dir, record["image"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Build target vector ordered by class index (alphabetical)
        scores = record["scores"]
        target_vector = torch.tensor(
            [
                scores.get(class_name, 0.0)
                for class_name in self.class_to_idx
            ],
            dtype=torch.float32
        )
        return image, target_vector
    
    def __len__(self):
        return len(self.records)


def build_class_index(jsonl_paths: list[str]) -> dict:
    """
    Scans all JSONL files to collect every unique class name,
    then assigns indices in sorted (alphabetical) order —
    matching ImageFolder's original behaviour.
    """
    class_names = set()
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    # Build Classes From scores
                    for class_name in record["scores"].keys():
                        class_names.add(class_name)                    

    sorted_names = sorted(class_names)
    return {name: idx for idx, name in enumerate(sorted_names)}

# if batch_size = 16, then images, targets = next(iter(train_loader)) gives:
# images.shape: torch.Size([16, 3, 224, 224])
# targets.shape: torch.Size([16, num_classes]) : torch.Size([16, 7]), for a 7-class waste dataset.
def get_dataloaders(batch_size=16, num_workers=2):       
        """
        Loads train from train.jsonl and val from val.jsonl.
        Returns dataloaders + class info.
        """

        class_to_idx = build_class_index([TRAIN_FILE, VAL_FILE])
        class_names  = list(class_to_idx.keys())
        num_classes  = len(class_names)
        idx_to_class = { idx: name for name, idx in class_to_idx.items()}

        train_dataset = JSONLDataset(TRAIN_FILE, DATA_DIR, class_to_idx, transform=train_transforms)
        val_dataset   = JSONLDataset(VAL_FILE,   DATA_DIR, class_to_idx, transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, class_names, num_classes, idx_to_class