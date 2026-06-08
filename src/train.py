####################################################################
#Step 3: Writing the training loop (epochs + batches + validation)
####################################################################

import torch
import torch.nn.functional as F

#forward > backward > update > return loss
def train_one_batch(model, images, labels, optimizer, criterion):
    # Forward pass
    outputs = model(images) #Pass the input images through the neural network and get its predictions. outputs.shape == (batch_size, num_classes); tensor([[ 2.3, -1.1,  0.5, ...],[ 0.7,  1.9, -0.2, ...]]); Each row = one image, Each column = one class. These are called logits (raw, unnormalized scores)
    log_probs = F.log_softmax(outputs, dim=1)   # KLDivLoss expects log-probs
    loss = criterion(log_probs, labels) #labels are plain probabilities, outputs: what the model predicts, loss: how wrong the model is

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return outputs, loss.item() # return log_probs for argmax accuracy

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    #for each batch
    for images, labels in dataloader:
        images = images.to(device) #images.shape = [batch_size, 3, 224, 224]
        labels = labels.to(device) #labels.shape = [batch_size, num_classes]

        outputs, loss = train_one_batch(model, images, labels, optimizer, criterion)

        # Statistics
        running_loss += loss * images.size(0)
        _, predicted = torch.max(outputs, 1)
        _, target_labels = torch.max(labels, 1)  # argmax of soft label vector
        correct += (predicted == target_labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=1)   # KLDivLoss expects log-probs
            loss = criterion(log_probs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            _, target_labels = torch.max(labels, 1)  # argmax of soft label vector
            correct += (predicted == target_labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    save_model_file,
    num_epochs=20
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss   
            print(f"best_val_loss: {best_val_loss}")         
            save_trained_model(model, optimizer, save_model_file) #save trained model weights for best val loss

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"|| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model, history


def save_trained_model(model, optimizer, filepath):
    torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict()
}, filepath)

def load_trained_model(model, filepath):
    checkpoint = torch.load(filepath)
    model_state_dict = checkpoint["model"]
    model.load_state_dict(model_state_dict)
    optimizer = checkpoint["optimizer"]
    return model, optimizer

#TESTCODE
#dummy training function, it confirms:
#Pretrained weights load
#Classifier head is correct
#Device handling is correct
#Forward pass works
def dummy_training1(model, images):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    print(outputs.shape) #Output shape: torch.Size([8, num_classes]), This confirms the model produces logits and that they can be converted to probabilities.


#TESTCODE
#dummy training function to test the training loop, it confirms:
#Loss function matches model output
#Gradients flow
#Optimizer works
#If it is runagain, will see a different loss, Random input → different logits, Random gradients → different update → slightly different output
def dummy_training2(model, device, optimizer, criterion):
    dummy_input = torch.randn(8, 3, 224, 224).to(device) #images.shape = [batch_size, 3, 224, 224], fakedata-creates random input images
    
    # hint: torch.randn() generates both positive and negative numbers.
    # torch.rand() generates values in:[0, 1], so every entry is non-negative.
    # tensor([
    # [0.25, 0.10, 0.05, 0.15, 0.20, 0.15, 0.10],
    # [0.05, 0.30, 0.10, 0.25, 0.10, 0.10, 0.10]
    # ])
    labels = torch.rand(8, 7).to(device) #labels.shape = [batch_size, num_classes]
    labels = labels / labels.sum(dim=1, keepdim=True)
    
    model.train() #sets the model in training mode, acttivates Dropout layers, BatchNorm updates, it is needed before forward + backward passes during training 
    optimizer.zero_grad() #clears any previous gradients stored in PyTorch’s computational graph
    outputs = model(dummy_input) #feeds the dummy batch through the model; produces a tensor of shape [8, num_classes], each row = logits for the 8 samples, logits are raw scores (before softmax)
    log_probs = F.log_softmax(outputs, dim=1)   # KLDivLoss expects log-probs
    loss = criterion(log_probs, labels) #computes the loss between model predictions and the provided labels, KLDivLoss compares the surrogate probability distribution against the target model probability distribution. Loss is higher when predictions are far from labels, lower when correct.
    loss.backward() #Computes gradients of the loss w.r.t all trainable parameters, PyTorch builds a computational graph on-the-fly and backpropagates, Prepares weights for the optimizer
    optimizer.step() #Updates all trainable parameters using the gradients, Effectively performs one training step on the dummy batch
    print("Loss:", loss.item()) #loss is just a measure of how “wrong” the model was on this random batch.Since both inputs and labels are random, Loss is meaningless for learning, it’s just a sanity check

#TESTCODE
def test_one_training_step(model, images, labels, optimizer, criterion):
    model.train()
    outputs1, loss1 = train_one_batch(model, images, labels, optimizer, criterion)
    outputs2, loss2 = train_one_batch(model, images, labels, optimizer, criterion)
    print(f"loss1 = {loss1}, loss2 = {loss2}")
    
