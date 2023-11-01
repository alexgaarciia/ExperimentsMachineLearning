import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

from code import models_run

# Load the Penguin and Abalone dataset
penguin_data = pd.read_csv('./penguins.csv')
abalone_data = pd.read_csv('./abalone.csv')

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
sex_mapping = {'FEMALE': 0, 'MALE': 1}
# These dictionaries serve as lookup tables where the keys represent the original ones but formatted
penguin_data['island'] = penguin_data['island'].map(island_mapping)
penguin_data['sex'] = penguin_data['sex'].map(sex_mapping)
# Now, penguin_data contains 'island' and 'sex' features in numerical format based on manual categorization.

# 1b: Determine if the Abalone dataset can be used as is; otherwise convert any features using the 2 methods above
# As all the columns are float or integer type, we can use Abalone as it is
abalone_data_types = abalone_data.dtypes
print(abalone_data_types)


# 2: Plot the percentage of the instances in each output class and store the graphic in a file called penguin-classes.gif

# Assuming the target in 'penguin_data' is 'species'
class_counts = penguin_data['species'].value_counts(normalize=True) * 100

plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Species')
plt.ylabel('Percentage of Instances')
plt.title('Percentage of Instances in Each Species (Penguin Dataset)')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('penguin-classes.png')

# Convert the PNG file to GIF using Pillow
img = Image.open('penguin-classes.png')
img.save('penguin-classes.gif', format='GIF')

plt.show()

# Assuming the target in 'abalone_data' is 'Type'
class_counts = abalone_data['Type'].value_counts(normalize=True) * 100

plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='lavender')
plt.xlabel('Type')
plt.ylabel('Percentage of Instances')
plt.title('Percentage of Instances in Each Type (Abalone Dataset)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('abalone-classes.png')

# Convert the PNG file to GIF using Pillow
img = Image.open('abalone-classes.png')
img.save('abalone-classes.gif', format='GIF')

plt.show()

# 3. Split the dataset using train test split using the default parameter values.
# Split the dataset into features (X) and target variable (y)
X = penguin_data[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = penguin_data['species']

models_run(X, y)  # run models for Pengins data

type_mapping = {'F': 0, 'M': 1, 'I': 2}
X_abalone = abalone_data[['LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']]
y_abalone = abalone_data['Type'].map(type_mapping)

models_run(X_abalone, y_abalone)

