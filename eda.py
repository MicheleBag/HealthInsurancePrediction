import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import preprocessing


DATASET_NAME = 'HealthInsurance - OriginalDataset.csv'
dataset = pd.read_csv(DATASET_NAME)
pd.set_option('display.max_columns', None)

print(dataset.info())
print(dataset.head())
# print(dataset.tail())

# Check NaN value
check_nan = dataset.isnull().values.any()
count_nan = dataset.isnull().sum().sum()
print('\n\nNaN values: ', check_nan)
print('NaN count: ', count_nan)

# Remove duplicated samples
print('\n\nDataset shape: ', dataset.shape)
duplicate_rows = dataset[dataset.duplicated()]
print('#Duplicated rows: ', duplicate_rows.shape)

# Dummyfying non numerical features
columns_to_dummy = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
dataset = pd.get_dummies(dataset, columns=columns_to_dummy)
print(dataset.head(10))

# Remove useless binary features
columns_to_drop = ['Gender_Female', 'Vehicle_Damage_No']
dataset.drop(columns_to_drop, axis=1, inplace=True)

# Rename complementary features of deleted ones
dataset.rename(columns={'Gender_Male': 'Gender', 'Vehicle_Damage_Yes': 'Vehicle_Damage'}, inplace=True)
print(dataset.info())

# Check for unbalanced features
columns_to_check = ['Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Policy_Sales_Channel',
                    'Gender', 'Vehicle_Damage', 'Vehicle_Age_1-2 Year',
                    'Vehicle_Age_< 1 Year', 'Vehicle_Age_> 2 Years', 'Response']
for idx, value in enumerate(columns_to_check):
    print("\n", value, "count: ", dataset[value].value_counts())
    print(value, "Normalized count: ", dataset[value].value_counts(normalize=True))
    plt.subplot(3, 4, idx+1)
    dataset[value].value_counts(normalize=True).plot.pie()


# Dropping features Driving_License because almost all sample has driving license:
# 1 = 0.998 && 0 = 0.002
dataset.drop('Driving_License', axis=1, inplace=True)

# Check collinearity between features
corrMatrix = dataset.corr()
plt.figure()
sn.heatmap(corrMatrix, annot=True)

# Checking if Annual_Premium is correlated with Age -> NO
plt.figure()
plt.scatter(dataset['Age'], dataset['Annual_Premium'])

# Boxplots
columns_to_boxplot = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
plt.figure()
dataset.boxplot(columns_to_boxplot)

# Checking for HLP for Annual_Premium feature
plt.figure()
dataset.boxplot('Annual_Premium')

# Removing 1% of the highest values
dataset = dataset[(dataset.Annual_Premium < dataset.Annual_Premium.quantile(.99))]
plt.figure()
dataset.boxplot('Annual_Premium')

# Features normalization
features_to_normalize = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
scaler = preprocessing.MinMaxScaler()
dataset[features_to_normalize] = scaler.fit_transform(dataset[features_to_normalize])
print(dataset.head())

plt.show()
dataset.to_csv('Dataset.csv', index=False)
