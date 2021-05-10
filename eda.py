import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import preprocessing

"""
TODO: 
    -finire controllo sul bilanciamento delle classi 
        1 = 0.12 && 0 = 0.88
    -provare senza auto maggiori a 2 anni
"""

DATASET_NAME = 'HealthInsurance - OriginalDataset.csv'
dataset = pd.read_csv(DATASET_NAME)
pd.set_option('display.max_columns', None)

print(dataset.info())
print(dataset.head())
# print(dataset.tail())

# Controllo se nel dataset sono presenti eventuali valori NaN da gestire
check_nan = dataset.isnull().values.any()
count_nan = dataset.isnull().sum().sum()
print('NaN values: ', check_nan)
print('NaN count: ', count_nan)

# Rimuovo eventuali campioni duplicati
print('Dataset shape: ', dataset.shape)
duplicate_rows = dataset[dataset.duplicated()]
print('#Duplicated rows: ', duplicate_rows.shape)

# Dummizzo la features non numeriche
columns_to_dummy = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
dataset = pd.get_dummies(dataset, columns=columns_to_dummy)
print(dataset.head(10))

# Rimuovo features binarie inutili
columns_to_drop = ['Gender_Female', 'Vehicle_Damage_No']
dataset.drop(columns_to_drop, axis=1, inplace=True)

# Rinomino features complementari a quelle rimosse
dataset.rename(columns={'Gender_Male': 'Gender', 'Vehicle_Damage_Yes': 'Vehicle_Damage'}, inplace=True)
print(dataset.info())

# Controllo il bilanciamento delle feature nel dataset
columns_to_check = ['Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Policy_Sales_Channel',
                    'Gender', 'Vehicle_Damage', 'Vehicle_Age_1-2 Year',
                    'Vehicle_Age_< 1 Year', 'Vehicle_Age_> 2 Years', 'Response']
for column in columns_to_check:
    #print(column, "count: ", dataset[column].value_counts())
    #print(column, "Normalized count: ", dataset[column].value_counts(normalize=True))
    plt.figure()
    dataset[column].value_counts(normalize=True).plot.pie()

# Rimuovo la feature Driving_License poichè praticamente tutti i campioni nel
# dataset hanno la patente di guida:
# 1 = 0.998 && 0 = 0.002
dataset.drop('Driving_License', axis=1, inplace=True)

# Controllo collinearità tra features
corrMatrix = dataset.corr()
plt.figure()
sn.heatmap(corrMatrix, annot=True)

# Controllo se il premio è correlato all'età dell'assicurato -> NO
plt.figure()
plt.scatter(dataset['Age'], dataset['Annual_Premium'])

# Boxplots
columns_to_boxplot = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
plt.figure()
dataset.boxplot(columns_to_boxplot)

# Verifico HLP per la feature Annual_Premium
plt.figure()
dataset.boxplot('Annual_Premium')

# Rimuovo l'1% dei valori più grandi
dataset = dataset[(dataset.Annual_Premium < dataset.Annual_Premium.quantile(.99))]
plt.figure()
dataset.boxplot('Annual_Premium')

# Features normalization
features_to_normalize = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
scaler = preprocessing.MinMaxScaler()
dataset[features_to_normalize] = scaler.fit_transform(dataset[features_to_normalize])
#print(dataset.head())


#plt.show()
dataset.to_csv('Dataset.csv', index=False)

