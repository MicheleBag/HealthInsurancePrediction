import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

"""
TODO:
- usa F1 score invece che Accuracy
- L1/L2 normalization?
"""

DATASET_NAME = 'Dataset.csv'
TEST_SIZE = .2
VALIDATION_SIZE = .2
dataset = pd.read_csv(DATASET_NAME)

#print(dataset.info())
# print(dataset.head())
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
neighbors_range = (1, 25)

for k in k_fold:
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, val_index in skf.split(X_train, y_train):
        scores_list = []
        print("TRAIN:", train_index, "VAL:", val_index)
        X_train2, X_val = X_train[train_index], X_train[val_index]
        y_train2, y_val = y_train[train_index], y_train[val_index]
        print(X_train2.shape, y_train2.shape, X_val.shape, y_val.shape)
        # KNN
        for n in neighbors_range:
            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(X_train2, y_train2)
            y_pred = knn.predict(X_val)
            scores_list.append(metrics.accuracy_score(y_val, y_pred))
        # Evaluation metrics
        plt.figure()
        plt.plot(neighbors_range, scores_list)
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        break

plt.show()


"""
# 1st try
# Model Selection
k_range = range(1, 25)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))

print(scores, scores_list)
plt.plot(k_range, scores_list)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
"""