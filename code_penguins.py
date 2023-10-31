import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the Penguin and Abalone dataset
penguin_data = pd.read_csv('/Users/lcsanchez/PycharmProjects/ExperimentsMachineLearning/penguins.csv')
abalone_data = pd.read_csv('/Users/lcsanchez/PycharmProjects/ExperimentsMachineLearning/abalone.csv')

# 1a:
# Method 1: Convert 'island' and 'sex' features into 1-hot vectors (dummy-coded data)
# Drop first category to avoid the 'dummy variable trap'
encoder = OneHotEncoder(sparse_output=False, drop='first')
# The fit_transform method fits the encoder to the specified columns and transforms the data simultaneously.
encoded_island_sex = encoder.fit_transform(penguin_data[['island', 'sex']])
# The resulting sparse matrix is transformed to a DataFrame
encoded_feature_names = encoder.get_feature_names_out(['island', 'sex'])
encoded_df = pd.DataFrame(encoded_island_sex, columns=encoded_feature_names)
# Add the one-hot transformed columns to the original data and drop the original 'sex' and 'penguin' columns
penguin_data_encoded_1hot = pd.concat([penguin_data, encoded_df], axis=1)
penguin_data_encoded_1hot.drop(['island', 'sex'], axis=1, inplace=True)

# Method 2: Convert 'island' and 'sex' features into categories manually
island_mapping = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
sex_mapping = {'Female': 0, 'Male': 1}
# These dictionaries serve as lookup tables where the keys represent the original ones but formatted
penguin_data['island'] = penguin_data['island'].map(island_mapping)
penguin_data['sex'] = penguin_data['sex'].map(sex_mapping)
# Now, penguin_data contains 'island' and 'sex' features in numerical format based on manual categorization.

# 1b: Determine if the Abalone dataset can be used as is; otherwise convert any features using the 2 methods above
# As all the columns are float or integer type, we can use Abalone as it is
abalone_data_types = abalone_data.dtypes
print(abalone_data_types)
