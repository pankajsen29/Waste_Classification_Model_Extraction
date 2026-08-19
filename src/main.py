
import src.config as cfg
DEBUG = False
PREDICT_QUERY = False # For: 1) obtaining the predictions in query_results.jsonl, 2) getting the results splitted into train.jsonl and val.jsonl

if DEBUG:
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    #NUM_EPOCHS = 10
    #BATCH_SIZE = 16
    #NUM_WORKERS = 0

else:
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    # NUM_EPOCHS = 20
    # BATCH_SIZE = 16
    # NUM_WORKERS = 2

if PREDICT_QUERY:
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
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)
#o/p: torch.Size([32, 3, 224, 224]) torch.Size([32, 7])
#images=a single batch pulled from train_loader
#labels=target score vector or the probability vector, e.g., tensor([0.25, 0.10, 0.05, 0.15, 0.20, 0.15, 0.10], [0.05, 0.30, 0.10, 0.25, 0.10, 0.10, 0.10],....)


######## Step 4: model initialization #########
from src.model import (
    get_model,
    get_loss_function,
    get_optimizer,
    get_device
)

device = get_device()
model = get_model(cfg.MODEL_NAME, num_classes)
model = model.to(device)

# KLDivLoss is needed as loss function because we have soft label probability vectors.
criterion = get_loss_function()

optimizer = get_optimizer(model, optimizer_name=cfg.OPTIMIZER_NAME, lr=cfg.LEARNING_RATE)


######### Step 5: train the model #######
import json
from src.train import (
    train_model, 
    save_trained_model,
    load_model_checkpoint,
    dummy_training1, 
    dummy_training2,
    test_one_training_step
)

#TESTCODE
#dummy_training1(model, images)

#TESTCODE: completely independent of the dataset.
#ensure optimizer, criterion are set
#dummy_training2(model, device, optimizer, criterion)

#TESTCODE
#test_one_training_step(model, images, labels, optimizer, criterion)


if DEBUG:    
    # 1. load checkpoint first
    checkpoint = load_model_checkpoint(cfg.MODEL_CHECKPOINT_FILE, device)

    # 2. load trained weights
    model.load_state_dict(checkpoint["model"])
    class_names = checkpoint["class_names"]
    optimizer = checkpoint["optimizer"]

    #loading the history from json
    with open(cfg.TRAINING_HISTORY_JSON, "r") as f:
        history = json.load(f)
else:
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        class_names,
        device,
        cfg.MODEL_CHECKPOINT_FILE,
        num_epochs=NUM_EPOCHS
    )

    #saving trained model weights is done during training based on best validation loss

    #saving the history to json
    with open(cfg.TRAINING_HISTORY_JSON, "w") as f:
        json.dump(history, f)

    #loading the history from json
    with open(cfg.TRAINING_HISTORY_JSON, "r") as f:
        history = json.load(f)

'''
######### plots ##############
from src.plots import plot_learning_curves

plot_learning_curves(history)

#how to interpret learning curves
#Train (Down), Val (Down) : Good fit
#Train (Down), Val (Up) :  Overfitting
#Both high & flat : Underfitting
'''

'''
############## tuning (OPTIONAL) ###################
OPTIONAL at beginning: same settings is used as the target model
    # Transforms
    # Batch Size
    # Optimizer
    # Learning Rate
    # Epochs

FUTURE experiments:
    # different architecture (EfficientNet-B0 vs ResNet18)
    # frozen vs fine-tuned backbone: feature_extract = True
    # Learning Rate: 0.001, 0.0005, 0.0001
'''