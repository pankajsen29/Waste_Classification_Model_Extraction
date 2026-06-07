############################################################
# Step 4 (during model extraction training): Model initialization 
# (ResNet18 + transfer learning)
############################################################

import torch
import torch.nn as nn
import torchvision.models as models


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ResNet18 feature size: 512
#manually downloaded weights from: (as the download doesn't happen due to SSL error in windows)
#https://download.pytorch.org/models/resnet18-f37072fd.pth
#and copied to: "C:\Users\panka\.cache\torch\hub\checkpoints"
#otherwise: model = models.resnet18(weights=None)
def get_resnet18(num_classes, feature_extract=True):
    """
    Initializes a pretrained ResNet18 (with transfer learning) for multi-class classification.

    Args:
        num_classes (int): number of output classes
        feature_extract (bool): True means -> use the pretrained network as a fixed feature extractor; in other words, freeze pretrained layers
        (feature_extract answers: Do I want to reuse pretrained features as-is, or do I want to adapt them by training?)
    Returns:
        model (nn.Module)
    """
    
    '''Loads ImageNet-pretrained weights; CNN backbone ready'''
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    '''Freeze pretrained layers (Optimizer updates only required layers) (feature extraction mode),
    This ensures: backbone weights are frozen, optimizer won't touch them,
    this prevents modifying learned ImageNet features.'''
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False #don't update these weights during backpropagation

    '''Custom classifier for waste categories,
       i.e., replace the final classification layer
       why: original fc = 1000 classes (ImageNet), our task = waste categories
    '''
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    #Ensuring classifier is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


#ResNet34 feature size: 512
#downloaded weights from: https://download.pytorch.org/models/resnet34-b627a593.pth
def get_resnet34(num_classes, feature_extract=True):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model

#EfficientNet-B0 feature size: 1280
#downloaded weights from: https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth
def get_efficientnet_b0(num_classes, feature_extract=True):
    
    #error: file name mismatch between what is downloaded and what pytorch expects
    #so renamed it to "efficientnet_b0_rwightman-7f5810bc.pth"
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    #alternative method: using the full path of the .pth
    #model = models.efficientnet_b0(weights=None)
    #state_dict = torch.load(r"C:\Users\panka\.cache\torch\hub\checkpoints\efficientnet_b0_rwightman-3dd342df.pth")
    #model.load_state_dict(state_dict)

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    for param in model.classifier[1].parameters():
        param.requires_grad = True

    return model

#NOT TESTED
#mobilenet_v2 feature size: 1280
#MobileNet is different:The classifier is inside model.classifier, It is a Sequential block
#Same idea, but different attribute name (classifier instead of fc)
#downloaded weights from: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
def get_mobilenet_v2(num_classes, feature_extract=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    for param in model.classifier[1].parameters():
        param.requires_grad = True

    return model

def get_model(model_name, num_classes, feature_extract=True):
    if model_name == "resnet18":
        return get_resnet18(num_classes, feature_extract) #primary - main CNN result
    if model_name == "resnet34":
        return get_resnet34(num_classes, feature_extract) #baseline
    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes, feature_extract) #best final model
    elif model_name == "mobilenet_v2":
        return get_mobilenet_v2(num_classes, feature_extract)
    else:
        raise ValueError("Unsupported model")


def get_loss_function():
    """
    Loss function for soft label probability vectors.

    KLDivLoss expects log-probabilities from the model, so need to wrap model output as below:
    outputs = F.log_softmax(model(images), dim=1)
    Also need to make sure the model does NOT have a softmax in its final layer

    reduction="batchmean": it divides by batch size, giving a per-sample average loss
    """
    return nn.KLDivLoss(reduction="batchmean")

#lr=1e-3
def get_optimizer(model, optimizer_name="adam", lr=0.001):
    """
    Optimizer that updates only trainable parameters (or only the classifier head (or unfrozen layers))
    """
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(params_to_update, lr=lr)

    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(params_to_update, lr=lr, momentum=0.9, weight_decay=0.0)

    else:
        raise ValueError("Unsupported optimizer")