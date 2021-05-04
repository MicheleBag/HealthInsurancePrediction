import pandas as pd
import matplotlib.pyplot as plt

DATASET_NAME = 'HealthInsurance - Dataset.csv'
dataset = pd.read_csv(DATASET_NAME)
print(dataset.head())

# Controllo se nel dataset sono presenti eventuali valori NaN da gestire
check_nan = dataset.isnull().values.any()
count_nan = dataset.isnull().sum().sum()
print('NaN values: ' + str(check_nan))
print('NaN count: ' + str(count_nan))

