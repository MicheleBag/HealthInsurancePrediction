# HealthInsurancePrediction

The [dataset](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction) is provided from an Health Insurance Company that want to know if customers could be interested in a vehicle insurance.

# Exploratory data analysis:
- Check NaN values
- Remove duplicate samples
- Dummify non-numerical features
- Remove useless binary features (created from dummify step)
- Check unbalanced features
- Check collinearity
- Remove high influence points
- Normalization

# Classification:
- Split train/test as 80/20 stratifying y label
- Dataset is a pretty unbalanced so I've applied SMOTE to oversample the train-set
- Applied Grid Search Cross Validation to find out the best hyperparameters in a given value space
- Fit 2 models using logistic regression and neural network


# Results

| Model | Best Hyper-parameters | Confusion matrix | Accuracy | AUC | F1-score | Precision | Recall 
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | {'Cs': 1, 'cv': 3} | [[39119 27111][220 9010]] | 0.637 | 0.822 | 0.397 | 0.249 | 0.976
| Neural Network | {'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50),'learning_rate': 'adaptive'} | [[46185 20045][1419 7811]] | 0.715 | 0.833 | 0.421 | 0.280 | 0.846

*Confusion matrix order is based on sklearn implementation: [[TN FP][FN TP]]
