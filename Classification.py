import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
"""
TODO:

"""

DATASET_NAME = 'Dataset.csv'
TEST_SIZE = .2
dataset = pd.read_csv(DATASET_NAME)

print(dataset.info())
#print(dataset.head())
dataset.drop('id', axis=1, inplace=True)

# Splitto il dataset in train e test
y = dataset['Response']
X = dataset.drop('Response', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1, stratify=y)
# Controllo se le etichette positive sono proporzionate tra train e test
print("Train/Test shape:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("Test set % of labels == 1:", y_test.sum()/y.sum()*100)

