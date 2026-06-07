
DEBUG = True

if DEBUG:
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    #BATCH_SIZE = 16
    #NUM_WORKERS = 0

else:
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    # BATCH_SIZE = 16
    # NUM_WORKERS = 2


######## Step 1: call query api ######
from src.extraction.query_api import get_query_results

query_result_file, query_status = get_query_results()
if query_status:
    print(f"\nQuery results is stored in {query_result_file}")
else:
    print(f"\nQuery results is not ready!")


######## Step 2: split dataset ######
from src.extraction.split_dataset import get_dataset_splitted

train_file, val_file = get_dataset_splitted()
print(f"\nTrain dataset file: {train_file}")
print(f"\nValidation dataset file: {val_file}")

######## Step 3: get dataloaders ######
from src.extraction.build_dataset import get_dataloaders

train_loader, val_loader, class_names, num_classes, idx_to_class = get_dataloaders(BATCH_SIZE, NUM_WORKERS)

#TESTCODE
images, targets = next(iter(train_loader))
print(images.shape, targets.shape)
#o/p: torch.Size([32, 3, 224, 224]) torch.Size([32, 7])
#images=a single batch pulled from train_loader
#targets=target vector, e.g., tensor([0.05, 0.04, 0.91])


######## Step 4: model initialization #########
from src.model import (
    get_model,
    get_loss_function,
    get_optimizer,
    get_device
)

device = get_device()
model = get_model("mobilenet_v2", num_classes) #lightweight model for training an extracted model
#model = get_model("resnet18", num_classes) #primary - main CNN result
#model = get_model("resnet34", num_classes) #baseline
#model = get_model("efficientnet_b0", num_classes) #best final model
model = model.to(device)

# KLDivLoss is needed as loss function because we have soft label probability vectors.
criterion = get_loss_function()

#optimizer = get_optimizer(model, optimizer_name="adam", lr=0.001)
optimizer = get_optimizer(model, optimizer_name="sgd", lr=0.001)
#optimizer = get_optimizer(model, optimizer_name="sgd", lr=0.01) #only with efficientnet_b0


