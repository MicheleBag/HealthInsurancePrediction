import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

"""
TODO:
- Precision&recall --> BAD RESULTS
- try different model?
- L1/L2 normalization?
"""

DATASET_NAME = 'Dataset.csv'
TEST_SIZE = .2
VALIDATION_SIZE = .2
dataset = pd.read_csv(DATASET_NAME)

#print(dataset.info())
pd.set_option('display.max_columns', None)
#print(dataset.head())
dataset.drop('id', axis=1, inplace=True)

# Splitto il dataset in training e test
y = dataset['Response'].values
X = dataset.drop('Response', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1, stratify=y)
# Controllo se le etichette positive sono proporzionate tra train e test
#print("Train/Test shape:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print("Test set % of labels == 1:", y_test.sum() / y.sum() * 100)

# Splitto il training set in training+validation set
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, random_state=1, stratify=y_train)


# Stratified K-Fold cross validation
k_fold = [5]  # , 10]
neighbors_range = range(1, 20)

for k in k_fold:
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    scores = [0]*(max(neighbors_range))
    precision = [0]*(max(neighbors_range))
    recall = [0]*(max(neighbors_range))

    for train_index, val_index in skf.split(X_train, y_train):
        print("TRAIN:", train_index, "VAL:", val_index)
        X_train2, X_val = X_train[train_index], X_train[val_index]
        y_train2, y_val = y_train[train_index], y_train[val_index]

        # KNN
        for n in neighbors_range:
            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(X_train2, y_train2)
            y_pred = knn.predict(X_val)
            #scores[n-1] += metrics.f1_score(y_val, y_pred, pos_label=1)
            precision[n-1] += metrics.precision_score(y_val, y_pred, pos_label=1)
            recall[n-1] += metrics.recall_score(y_val, y_pred, pos_label=1)
            #print(scores)
            # fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred, pos_label=1)
            #metrics.plot_confusion_matrix(knn, X_val, y_val)
            #metrics.plot_roc_curve(knn, X_val, y_val)
            #plt.show()

    # Evaluation metrics
    # print(fpr, tpr, thresholds)

    #scores = np.array(scores)
    #scores = scores / k_fold
    #print(scores)
    # plt.figure()
    # plt.plot(neighbors_range, scores)
    # plt.xlabel('K')
    # plt.ylabel('F1-Score')

    precision = np.array(precision)
    recall = np.array(recall)
    precision = precision / k_fold
    recall = recall / k_fold
    print(precision)
    print(recall)
    plt.figure()
    plt.plot(neighbors_range, precision)
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.figure()
    plt.plot(neighbors_range, recall)
    plt.xlabel('K')
    plt.ylabel('Recall')

plt.show()
