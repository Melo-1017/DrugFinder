# DrugFinder
The identification and discovery of druggable proteins have always been the core of drug development. Traditional structure-based identification methods are time-consuming and costly. As a result, more and more researchers have shifted their attention to sequence-based methods for identifying druggable proteins. This paper proposes a sequence-based druggable protein identification model called DrugFinder. The model extracts feature from the embedding output of the pre-trained protein model Prot_T5_Xl_Uniref50 and the evolutionary information of the PSSM matrix. Afterwards, it uses the random forest method to select features, and the selected features are trained and tested on multiple different machine learning classifiers. Among these classifiers, eXtreme Gradient Boosting (XGB) achieved the best results. The model using XGB achieved an accuracy of 94.98%, sensitivity of 96.33%, and specificity of 96.83% on an independent test set, which is better than existing identification methods.
