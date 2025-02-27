# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Load the data set
# data_url = "adult.data"
# df = pd.read_csv(data_url)

# # Data cleaning and conversion
# # ... Perform necessary data cleaning and conversion steps here ...

# # Split the data into features and target variable
# X = df.drop('target_variable_name', axis=1)  # Replace 'target_variable_name' with the actual target variable column name
# y = df['target_variable_name']

# # Data normalization
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Model training
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)

# # Evaluation
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


# Import necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# import matplotlib.pyplot as plt

# # Load your data
# df = pd.read_csv('adult.data')
# # Data Cleaning
# df = df.dropna() # remove missing values
# # Print the column names
# print(df.columns)
# # Print the first few rows of the DataFrame
# print(df.head())
# df['State-gov'] = df['State-gov'].map({'<=50K': 0, '>50K': 1}).astype(int) # convert income to numerical
# # Data Normalization
# x = df.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)
# # Data Preparing for Machine Learning Modeling
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Choosing Machine Learning Methods
# model = LogisticRegression()
# # Training
# model.fit(X_train, y_train)
# # Prediction
# y_pred = model.predict(X_test)
# # Evaluation
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# Required Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Data Visualization
# df = df.dropna()
sns.pairplot(df, hue='target')
plt.show()

# Data Cleaning
  # remove missing values

# Data Conversion
df['feature1'] = df['feature1'].astype(float)  # convert feature1 to float

# Data Normalization
scaler = preprocessing.StandardScaler()
df['feature1'] = scaler.fit_transform(df[['feature1']])

# Data Preparing for Machine Learning Modeling
X = df[['feature1', 'feature2']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Choosing Machine Learning Methods
model = LogisticRegression()

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
