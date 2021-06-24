import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, \
    roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# Sklearn design matrix
#   TN  FP
#   FN  TP


def model_scores(y_test, y_pred, y_pred_proba, show_plot):
    # Evaluation metrics
    print('Confusion matrix :\n', confusion_matrix(y_test, y_pred))
    print('Accuracy :', accuracy_score(y_test, y_pred))
    print('AUC score :', roc_auc_score(y_test, y_pred_proba))
    print('F1-score :', f1_score(y_test, y_pred))
    print('Precision score :', precision_score(y_test, y_pred))
    print('Recall score :', recall_score(y_test, y_pred))
    # ROC
    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_proba)
    if show_plot:
        plt.subplot(1, 2, 2)
        plt.title('ROC-AUC: test set')
    else:
        plt.subplot(1, 2, 1)
        plt.title('ROC-AUC: training set')
    plt.plot(fpr_test, fpr_test)
    plt.plot(fpr_test, tpr_test)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if show_plot:
        plt.figure()


def model_evaluation(model, X_train, y_train, X_test, y_test, trained=False):
    if not trained:
        model.fit(X_train, y_train)
    # Predict on train set
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[::, 1]
    print('Train dataset :')
    model_scores(y_train, y_pred, y_pred_proba, show_plot=False)
    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    print('\nTest dataset :')
    model_scores(y_test, y_pred, y_pred_proba, show_plot=True)


DATASET_NAME = 'Dataset.csv'
TEST_SIZE = .2
dataset = pd.read_csv(DATASET_NAME)

pd.set_option('display.max_columns', None)
print(dataset.head())
dataset.drop('id', axis=1, inplace=True)

# Split dataset into train and test set
y = dataset['Response'].values
X = dataset.drop('Response', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1, stratify=y)
# Check if label are balanced
# print("Train/Test shape:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print("Test set % of labels == 1:", y_test.sum() / y.sum() * 100)
# Label are balanced

# Logistic Regression
print("\n##### LOGISTIC REGRESSION #####")
parameter_space = {
    'Cs': [1, 3, 5, 10],
    'cv': [3, 5]
}
log_reg = LogisticRegressionCV(parameter_space)
clf = GridSearchCV(log_reg, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, y_train)
print('Best parameters found:\n', clf.best_params_)
model_evaluation(clf, X_train, y_train, X_test, y_test, trained=True)


# Dataset oversampling
smote = SMOTE(sampling_strategy='minority')
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
#  Check if label are balanced
# print("Oversampled train shape:", X_train_sm.shape, y_train_sm.shape)
# print("Original train shape:", X_train.shape, y_train.shape)
# print("#1 in oversampled train:", y_train_sm.sum())
# print("#1 in original train:", y_train.sum())
# Labels are equally distributed

# Logistic Regression with oversampled train
print("\n##### LOGISTIC REGRESSION (oversampled data) #####")
clf = GridSearchCV(log_reg, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train_sm, y_train_sm)
print('Best parameters found:\n', clf.best_params_)
model_evaluation(clf, X_train_sm, y_train_sm, X_test, y_test, trained=True)


# Neural network parameters
parameter_space = {
    'hidden_layer_sizes': [(8, 5), (7,), (10, 3), (30,), (50, 50, 50), (50, 100, 50), (100,)],
    'alpha': [0.0001, 0.1, 1, 3, 5],
    'learning_rate': ['constant', 'adaptive'],
}

# Neural network
print("\n##### NEURAL NETWORKS #####")
# mpl = MLPClassifier(max_iter=500)
# Searching best parameters
# clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
# clf.fit(X_train, y_train)
# Best parameter set
# print('Best parameters found:\n', clf.best_params_)
# {'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate': 'adaptive'}
# model_evaluation(clf, X_train, y_train, X_test, y_test, trained=True)
mlp = MLPClassifier(max_iter=500, alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='adaptive')
model_evaluation(mlp, X_train, y_train, X_test, y_test, trained=False)

# Neural network with oversampled train
print("\n##### NEURAL NETWORKS (oversampled data) #####")
# mpl = MLPClassifier(max_iter=500)
# Searching best parameters
# clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
# clf.fit(X_train_sm, y_train_sm)
# Best parameter set
# print('Best parameters found:\n', clf.best_params_)
# {'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50), 'learning_rate': 'adaptive'}
# model_evaluation(clf, X_train_sm, y_train_sm, X_test, y_test, trained=True)
mlp = MLPClassifier(max_iter=500, alpha=0.0001, hidden_layer_sizes=(50, 100, 50), learning_rate='adaptive')
model_evaluation(mlp, X_train_sm, y_train_sm, X_test, y_test, trained=False)

plt.show()
