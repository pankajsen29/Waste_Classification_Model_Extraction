### Links of all related projects:

**1. Target Project: [Waste_Classification_Multiclass_CNN_Pytorch]**
**https://github.com/pankajsen29/Waste_Classification_Multiclass_CNN_Pytorch**

> this is the main multiclass waste image classifier model which is used as the target/victim by the attacker/user, whose weights and parameters are hidden, but it returns prediction results to user queries. With these results attacker intends to build a copycat model, which is termed as surrogate model.


**2. Surrogate (i.e., extracted model) Project: [Waste_Classification_Model_Extraction]**  [ -> CURRENT REPOSITORY]
**https://github.com/pankajsen29/Waste_Classification_Model_Extraction**

> this is the copycat model attacker builds using the prediction results obtained via querying the target model.


**3. Evaluation (of extracted model) Project: [Waste_Classification_Model_Extraction_Evaluation]** 
**https://github.com/pankajsen29/Waste_Classification_Model_Extraction_Evaluation**

> this is the independent evaluation project for the extracted waste classification model which evaluates: 
>
>i) Fidelity or Agreement rate: Target Prediction vs Surrogate Prediction,
>
>ii) confidence similarity: Validation KL Divergence (Target score vector vs Surrogate score vector) etc.

# Waste_Classification_Model_Extraction

## 1. INTRODUCTION

This project:
* Trains a surrogate model using API predictions and confidence scores received from a target model. 
* Evaluates how closely the surrogate mimics the target model without access to its internals. 

It demonstrates:
* the black-box attack (having access only to input/output) on a waste image classification model,
* reconstructing a waste classification model (i.e., the surrogate model) with the prediction results (along with confidence scores) obtained via prediction queries to another model (i.e., the target),
* security risks of exposing ML models through prediction APIs,
* model extraction attacks can approximate a target model without access to its weights or architecture.
  > - black-box access ≠ complete security,
  > 
  > - it is very common attack without any kind of access to the details of the target model,
* extracted models may threaten the intellectual property of deployed AI systems.
  > - therefore, understanding model extraction risks helps improve the security of AI deployments.


Let's decode few of the jargons first:

**BLACK BOX ATTACK:**

* Attacker sees only: 
  - input and prediction output

* And does NOT see:
  - model weights,
  - training data,
  - architecture,
  - source code. 

* Suppose a waste image classifier predicts:
  - Input: bottle.jpg
  - Output: Plastic (92%)

* The attacker can repeatedly do this:
  - query(image) > prediction

* That alone is sufficient for a model extraction attack. 


**PREDICTION QUERY ACCESS:**

* It simply means:
  - Someone can give your model a new input and receive the model’s prediction output.
  - That’s all.

* NOT of any use:
  - a special training setting,
  - a special ML algorithm,
  - or anything enabled during training.
 
## 2. PIPELINE

<img width="1312" height="577" alt="image" src="https://github.com/user-attachments/assets/f25d68a9-92b8-4185-b811-4985bb0afd88" />

## 3. DATASET: TARGET VS SURROGATE

<img width="1339" height="581" alt="image" src="https://github.com/user-attachments/assets/3161cdb8-1753-41f0-84d4-87cb30d13a11" />

**Why class-mismatch is not an issue?**

**Pure Model Extraction = mimicking the target model**

* Hence, class mis-match is not an issue here, because:
  - The surrogate’s job is just to learn the mapping by: Image > Target Model > get prediction
  - It is trying to learn the behaviour of the target model.
  - Also it is not trying to improve the classification.

* the surrogate is not trying to learn the TrashBox labels. 
  - It is not training a new classifier using TrashBox annotations.
  - Therefore, the original TrashBox class is not required for training.

* Though for future the “true_label” is also stored to analyse:
  - Which TrashBox classes map to which RealWaste classes.
  - Confusion patterns.
  - Label distribution.
 
## 3. DATA PREPROCESSING: TARGET VS SURROGATE

**Data preprocessing steps for training are same for both target and surrogate:**

* Image: RGB
  
* Input image size: 524 x 524

* Resized to: 224 x 224

* Augmentation (only for training set): 
  - Resize, crop, flip, rotation
  - Brightness, contrast, saturation
  - Normalization (using the mean & std of ImageNet dataset)
  - No augmentation on test/validation set (only resize + normalization)

* Dataset (RealWaste) splits (Target): train (70%) + validation(15%) + test(15%)
  
* Dataset (TrashBox) splits (Surrogate): train (80%) + validation(20%)

## 4. MODEL ARCHITECTURE: TARGET VS SURROGATE

**TARGET:**

* Pre-trained weights: **RestNet18** (Transfer Learning)
* Architecture: **Heavier** (compared to MobileNet_V2)
* Parameters: **11.7M** (More)


**SURROGATE:**

* Pre-trained weights: **MobileNet_V2** (Transfer Learning)
* Architecture: 
  - **Lighter**: based on publicly available comparison. 
  - Different: which demonstrates model behavior can be stolen even without knowing architecture.
  - Remark: Even a lightweight model can successfully imitate a larger proprietary model.
  - Why not smaller custom CNN?: lower extraction accuracy likely.
  - Transfer learning is used assuming the target model is also built with transfer learning – easy guess
* Parameters: **3.4M** (fewer)
  - Computation: lower
  - Training: faster

## 5. TRAINING: TARGET VS SURROGATE

**TARGET:**

* Loss Function: **CrossEntropyLoss**
  - (Hint: model o/p > logits > loss computation)
  - (because hard labels are compared) – from dataset

* Optimizer: **Adam**

* Settings: 
  - NUM_EPOCHS = 10, 
  - BATCH_SIZE = 32, 
  - NUM_CLASSES = 9  (HARD), 
  - Learning Rate: 0.001

* Training loop (for each epoch, each batch): 
  - forward pass > loss computation > back propagation > weight updates

* Validation loop (for each epoch, each batch): 
  - includes only forward pass

**SURROGATE:**

* Loss Function: **KLDivLoss**
  - (Hint: model o/p > logits > softmax (to probability vector) > loss computation)
  - (because soft labels or the probability vector is compared) – from prediction query results

* Optimizer: **Adam**

* Settings: 
  - NUM_EPOCHS = 10, 
  - BATCH_SIZE = 32, 
  - NUM_CLASSES = 9  (SOFT), 
  - Learning Rate: 0.001

* Training loop (for each epoch, each batch): 
  - forward pass > loss computation > back propagation > weight updates

* Validation loop (for each epoch, each batch): 
  - includes only forward pass

**Training settings and output for all the epochs:**

**Model: mobilenet_v2, Loss function: kldivLoss, Optimizer: adam, learning rate: 0.001, Epoch: 10, Batch:32**

        Epoch [1/10]
        Train Loss: 0.5887 | Train Acc: 0.5777 || Val Loss: 0.4225 | Val Acc: 0.6374
        best_validation_loss: 0.422467137589065
        Model is saved as current_validation_loss < best_validation_loss
        
        Epoch [2/10]
        Train Loss: 0.5359 | Train Acc: 0.5993 || Val Loss: 0.4231 | Val Acc: 0.6436
        Model is not saved as current_validation_loss > best_validation_loss
        
        Epoch [3/10]
        Train Loss: 0.5178 | Train Acc: 0.6049 || Val Loss: 0.4289 | Val Acc: 0.6315
        Model is not saved as current_validation_loss > best_validation_loss
        
        Epoch [4/10]
        Train Loss: 0.5138 | Train Acc: 0.6078 || Val Loss: 0.4187 | Val Acc: 0.6366
        best_validation_loss: 0.4186742536288755
        Model is saved as current_validation_loss < best_validation_loss
        
        Epoch [5/10]
        Train Loss: 0.5178 | Train Acc: 0.6044 || Val Loss: 0.4218 | Val Acc: 0.6490
        Model is not saved as current_validation_loss > best_validation_loss
        
        Epoch [6/10]
        Train Loss: 0.5145 | Train Acc: 0.6089 || Val Loss: 0.4432 | Val Acc: 0.6397
        Model is not saved as current_validation_loss > best_validation_loss
        
        Epoch [7/10]
        Train Loss: 0.5158 | Train Acc: 0.6104 || Val Loss: 0.4480 | Val Acc: 0.6393
        Model is not saved as current_validation_loss > best_validation_loss
        
        Epoch [8/10]
        Train Loss: 0.5109 | Train Acc: 0.6126 || Val Loss: 0.4167 | Val Acc: 0.6498
        best_validation_loss: 0.4167148387849563
        Model is saved as current_validation_loss < best_validation_loss
        
        Epoch [9/10]
        Train Loss: 0.5073 | Train Acc: 0.6141 || Val Loss: 0.4277 | Val Acc: 0.6475
        Model is not saved as current_validation_loss > best_validation_loss
        
        Epoch [10/10]
        Train Loss: 0.5155 | Train Acc: 0.6020 || Val Loss: 0.4229 | Val Acc: 0.6424
        Model is not saved as current_validation_loss > best_validation_loss

## 6. SAVING OF THE TRAINED MODEL STATE

* Tracked:
  - Training Accuracy
  - Training Loss
  - Validation Accuracy
  - Validation Loss

* Model is saved based on:
  - Lowest Validation Loss
  - rather than highest validation accuracy.
  - Why:
    - Accuracy only checks whether the top class matches.
    - KL divergence measures how well the full distribution is matched.
  - Example:
  
    <img width="389" height="83" alt="image" src="https://github.com/user-attachments/assets/8258235b-59a2-4f74-8c75-8871964c4a20" />
  
  - Accuracy: Both are equal.
  - But KL divergence says: Model B reproduces the target probabilities much better.
  - Since my goal is to mimic target model behavior, the lower KL loss is the better surrogate.


**Note: Trained models can be found inside project's "checkpoints" directory**

## 7. PREDICTION QUERY RESULTS

**Query Result JSON:**

    {
      "image": "train/plastic/image_01.jpg",
      "predicted_class": “Cardboard",
      "scores": 
       {
         "Cardboard": 0.9753, 
         "Food Organics": 0.0003, 
         "Glass": 0.0003, 
         "Metal": 0.0015, 
         "Miscellaneous Trash": 0.0021, 
         "Paper": 0.0153, 
         "Plastic": 0.0004, 
         "Textile Trash": 0.003, 
         "Vegetation": 0.0018
      }
      "true_class": “cardboard"
    }

**Why confidence score is crucial?**

* **Without confidence score:**
    - ImageA: “plastic”, ImageA: “paper”
    - This gives the attacker:
      - only the final decision,
      - no information about how confident the model is,
      - no information about “second-best” classes.
    - Information Leakage Level: Very limited.
    - Attacker only learns: Input X belongs to class Y
    - This makes extraction: slower, less accurate, requires many more queries.

* **With confidence score, attacker learns:**
    - final prediction,
    - relative class relationships,
    - decision boundary behavior.
    - This makes extraction: more accurate, requires fewer queries
    - Example: (imageB is close to the decision boundary)
      
          Image 		Plastic 		Paper 		Glass
          ImageA		0.99		    0.01		  0.00
          ImageB		0.52		    0.45		  0.03

**Note: Sample prediction query result JSON can be found inside project's "data/Query_Results" directory**


## 8. DEFENSES

Surrogate accuracy differs based on what the prediction query exposes (this is due for future experiments).

Hence few defense strategies for the production API would be:
* truncate probabilities (or the confidence scores), 
* return only top-k labels, 
* or return only the predicted class.


## 9. REFERENCES

RealWaste Dataset: 

https://github.com/sam-single/realwaste

https://www.mdpi.com/2078-2489/14/12/633

TrashBox Dataset: https://www.kaggle.com/datasets/saimonv/trashbox

ResNet: https://arxiv.org/abs/1512.03385

Mobilenet_V2: https://arxiv.org/abs/1801.04381

Pre-trained weights:

ResNet: https://download.pytorch.org/models/resnet18-f37072fd.pth

Mobilenet_V2: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

PyTorch: https://docs.pytorch.org/vision/0.12/models.html
