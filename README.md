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

>i) Fidelity or Agreement rate: Target Prediction vs Surrogate Prediction,

>ii) confidence similarity: Validation KL Divergence (Target score vector vs Surrogate score vector) etc.

# Waste_Classification_Model_Extraction
Waste classification model extraction project which demonstrates black-box attack. It includes the steps of training of a surrogate model via prediction queries to another target model of waste classification (source repo: "Waste_Classification_Multiclass_CNN_Pytorch"). 
