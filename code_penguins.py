# Import necessary libraries/modules
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from code import models_run
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# Define class needed to store information from the output
class FileOutput:
    def __init__(self, file_name):
        self.file = open(file_name, 'w')
        self.stdout = sys.stdout
        sys.stdout = self
        self.suppress_output = False  # New flag to control output

    def write(self, text):
        if not self.suppress_output:
            self.file.write(text)
            self.stdout.write(text)

    def flush(self):
        if not self.suppress_output:
            self.file.flush()
            self.stdout.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


# Load the Penguin and Abalone dataset
penguin_data = pd.read_csv('./penguins.csv')
abalone_data = pd.read_csv('./abalone.csv')


########################################################################################################################
# EXERCISE 1
########################################################################################################################
# 1A) METHOD 1: Convert 'island' and 'sex' features into 1-hot vectors (dummy-coded data)
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


# 1A) METHOD 2: Convert 'island' and 'sex' features into categories manually
# These dictionaries serve as lookup tables where the keys represent the original ones but formatted
island_mapping = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
sex_mapping = {'FEMALE': 0, 'MALE': 1}

# Now, penguin_data contains 'island' and 'sex' features in numerical format based on manual categorization.
penguin_data['island'] = penguin_data['island'].map(island_mapping)
penguin_data['sex'] = penguin_data['sex'].map(sex_mapping)


# 1B)
abalone_data_types = abalone_data.dtypes
print(abalone_data_types)


########################################################################################################################
# EXERCISE 2
########################################################################################################################
# PENGUIN DATASET:
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

# Show the graph
plt.show()


# ABALONE DATASET:
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

# Show the graph
plt.show()


########################################################################################################################
# EXERCISE 3
########################################################################################################################
# PENGUIN DATASET:
# Split the dataset into features (X) and target variable (y)
X_penguin = penguin_data[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y_penguin = penguin_data['species']
X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin = train_test_split(X_penguin, y_penguin)

# ABALONE DATASET:
X_abalone = abalone_data[['LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']]
y_abalone = abalone_data['Type']
X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone = train_test_split(X_abalone, y_abalone)


########################################################################################################################
# EXERCISE 4, 5
########################################################################################################################
# Run the model for the penguin dataset:
penguin_file = FileOutput('penguin-performance.txt')
models_run(1, X_penguin, X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin)
penguin_file.close()

# Run the model for the abalone dataset:
#abalone_file = FileOutput('abalone-performance.txt')
#models_run(1, X_abalone, X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone)
#abalone_file.close()


########################################################################################################################
# EXERCISE 6
########################################################################################################################
# Run the model for the penguin dataset:
penguin_file_ex6 = FileOutput('penguin-performance-5times.txt')
models_run(5, X_penguin, X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin)
penguin_file_ex6.close()

# Run the model for the abalone dataset:
#abalone_file_ex6 = FileOutput('abalone-performance-5times.txt')
#models_run(5, X_abalone, X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone)
#abalone_file_ex6.close()

