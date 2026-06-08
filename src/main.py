
DEBUG = False
PREDICT_QUERY = False

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


######### Step 5: train the model #######
from src.train import (
    train_model, 
    save_trained_model,
    load_trained_model,
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


#for saving the model state
from pathlib import Path
save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)
save_model_file = save_dir / "extracted_waste_seg_full_training_state.pth"
train_history_json = save_dir / "extracted_waste_seg_training_history.json"

if DEBUG:
    #existing model needs to passed, to load the saved weights from disc
    model, optimizer = load_trained_model(model, save_model_file)
    
    #loading the history from json
    with open(train_history_json, "r") as f:
        history = json.load(f)
else:
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        save_model_file,
        num_epochs=NUM_EPOCHS
    )

    #saving trained model weights is done during training based on best validation loss

    #saving the history to json
    import json    
    with open(train_history_json, "w") as f:
        json.dump(history, f)

    #loading the history from json
    with open(train_history_json, "r") as f:
        history = json.load(f)

