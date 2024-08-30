# KiteDDI
## A knowledge graph integrated Transformer model for Drug-Drug Interaction prediction

It is a common practice in modern medicine to prescribe multiple medications simultaneously
to treat diseases. However, these medications could have adverse reactions between them, known as Drug-
Drug Interactions (DDI), which have the potential to cause significant bodily injury and could even be fatal.
Hence, it is essential to identify all the DDI events before prescribing multiple drugs to a patient. Most
contemporary research for predicting DDI events relies on either information from Biomedical Knowledge
graphs (KG) or drug SMILES, with very few managing to merge data from both to make predictions.
While others use heuristic algorithms to extract features from SMILES and KGs, which are then fed into
a Deep Learning framework to generate output. In this study, we propose a KG-integrated Transformer
architecture to generate an end-to-end fully automated Machine Learning pipeline for predicting DDI events
with high accuracy. The algorithm takes full-scale molecular SMILES sequences of a pair of drugs and a
biomedical KG as input and predicts the interaction between the two drugs with high precision. The results
show superior performance in two different benchmark datasets compared to existing state-of-the-art models
especially when the test and training sets contain distinct sets of drug molecules. This demonstrates the
strong generalization of the proposed model, indicating its potential for DDI event prediction for newly
developed drugs. The model does not depend on heuristic models for generating embeddings and has a
minimal number of hyperparameters, making it easy to use while demonstrating outstanding performance
in low-data scenarios.

The key contributions and novelty of this work are outlined below:
* Superior performance and accuracy compared to other related state-of-the-art models especially on predictions in the inductive dataset split setting showing good generalization.
* It is an End-to-end machine learning model with no heuristic components that rely on domain expert knowledge.
* The proposed model is lightweight, computationally inexpensive, and easy to use as it is an end-to-end pipeline requiring very few hyperparameter optimizations.
* The proposed algorithm only requires 2 inputs (KG and SMILES) as opposed to the main benchmark model which requires five (KG, SMILES, MPNN, AFP, WEAVE). These other inputs need to be generated using separate full-scale algorithms raising the dependency of the method on other models.
* The proposed architecture shows better performance at low data settings compared to the main benchmark method.

![Alt text](./Full_model_cropped.jpg?raw=true "Block diagram for the full model architecture.")

## Execution Instructions:
>> Download the following Python packages in the environment:
* numpy==1.26.4
* pandas==2.2.2
* torch==2.2.2+cu121
* tqdm==4.66.2
* rdkit-pypi==2022.9.5
* scikit-learn==1.5.1

>> Download the dataset and model files from the following link:
>> Execute the following command in the terminal
