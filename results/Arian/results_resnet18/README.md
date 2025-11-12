# All the results are also visualized in the Google Colab file. Please check the training codes.

## Confusion matrix & accuracy:
Almost all predictions are correct. Only 1 sample out of 2425 was misclassified. This shows that the model learned the patterns distinguishing Real vs. Fake extremely well on this test set.

## Precision and Recall
Precision: Out of all samples predicted as a class, how many were actually that class.  
Precision for Real: 1.0 → all predicted Real samples were truly Real.  
Precision for Fake: 0.9991 → almost all predicted Fakes were truly Fake.  
Recall: Out of all actual samples of a class, how many were correctly predicted.  
Recall for Real: 0.9992 → almost all Real samples were correctly identified.  
Recall for Fake: 1.0 → all Fake samples were correctly identified.  
F1-score: Harmonic mean of precision and recall; values ~0.9996 indicate excellent balance.

## Equal Error Rate (EER)
EER = 0.08% → the point where false positive rate = false negative rate. Extremely low EER means the model is almost perfect in distinguishing classes.

ROC AUC: ROC AUC = 1.0 → perfect discrimination between Real and Fake. The model’s predicted scores separate the classes completely.

## Test Loss
Test loss ≈ 0.002 → small numerical error in the predictions, consistent with extremely high accuracy.
