import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

"""
TODO: 
    -check outlier (anche prima e dopo la rimozione dal dataset)
    -finire controllo sul bilanciamento delle features (occhio alla Response)
    -drop colonne binarie Gender e Vehicle_Damage
"""
DATASET_NAME = 'HealthInsurance - Dataset.csv'
dataset = pd.read_csv(DATASET_NAME)
pd.set_option('display.max_columns', None)

print(dataset.info())
print(dataset.head())
#print(dataset.tail())

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
print(dataset.head(15))

# Controllo collinearità tra features
"""
corrMatrix = dataset.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

# Plots
columns_to_boxplot = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
dataset.boxplot(columns_to_boxplot)
plt.show()
"""

# Il dataset contiene praticamente il 99% di campioni in cui Driving_License=1
columns_to_check = ['Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Policy_Sales_Channel',
                   'Response', 'Gender_Male', 'Vehicle_Damage_Yes', 'Vehicle_Age_1-2 Year',
                   'Vehicle_Age_< 1 Year', 'Vehicle_Age_> 2 Years']
for column in columns_to_check:
    print(column, "count: ", dataset[column].value_counts())
    print(column, "Normalized count: ", dataset[column].value_counts(normalize=True))
    dataset[column].value_counts(normalize=True).plot.pie()
    plt.show()


"""
# Controllo se il premio è correlato all'età dell'assicurato -> NO
plt.scatter(dataset['Age'], dataset['Annual_Premium'])
plt.show()

# Controlle se l'età è correlata al fatto di avere un auto nuova -> NO
plt.scatter(dataset['Age'], dataset['Vehicle_Age_< 1 Year'])
plt.show()
"""



