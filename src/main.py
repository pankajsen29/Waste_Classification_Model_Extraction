
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
#o/p: torch.Size([832, 3, 224, 224]) torch.Size([16, 7])
#images=a single batch pulled from train_loader
#targets=targets for that batch
