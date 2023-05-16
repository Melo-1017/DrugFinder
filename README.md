# DrugFinder

**Author:** Mu Zhang | Fengqiang Wan 

**Notes:** This repository presents a model to identify druggable proteins. The model is implemented in Python and its principles are described in detail in our paper. If you have any questions about its use or understanding, please contact us at this email address: zhang.mu@foxmail.com

**Abstract:** The identification and discovery of druggable proteins have always been the core of drug development. Traditional structure-based identification methods are time-consuming and costly. As a result, more and more researchers have shifted their attention to sequence-based methods for identifying druggable proteins. This paper proposes a sequence-based druggable protein identification model called DrugFinder. The model extracts feature from the embedding output of the pre-trained protein model Prot_T5_Xl_Uniref50 and the evolutionary information of the PSSM matrix. Afterwards, it uses the random forest method to select features, and the selected features are trained and tested on multiple different machine learning classifiers. Among these classifiers, eXtreme Gradient Boosting (XGB) achieved the best results. The model using XGB achieved an accuracy of 94.98%, sensitivity of 96.33%, and specificity of 96.83% on an independent test set, which is better than existing identification methods.

**Usage:**

1. Generate pre-trained embedding features using the pretrain.py file. This data will be saved in the clsdata folder.
2. Generate PSSM features using BLAST. We provide a set of PSSM features files generated using the uniref50 database in the pssmdata folder, you can also use other comparison databases to generate PSSM data as needed.
3. Use the deal.py file to process PSSM data and pre-trained embedding data. After that, the druggable protein identification results can be obtained.

**Requirement:**

- python==3.10
- torch==2.0.0+cu118
- scikit-learn==1.2.2
- bio-embeddings==0.2.2
- seaborn==0.12.2
- matplotlib==3.7.1
- numpy==1.22.4
- pandas==1.5.3
