import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report

# Step 1: Load the Dataset
data = pd.read_csv('adult.csv')

# Step 2: Data Preprocessing
# TODO: Handle missing values, encode categorical variables, etc.

# Step 3: Feature Extraction
# TODO: Select relevant features and convert categorical variables if needed

# Step 4: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split('Cross', 'validator', test_size=0.2, random_state=42)

# Step 5: Train the MultinomialNB Classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Step 6: Train the GaussianNB Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Step 7: Evaluate the Classifiers
mnb_predictions = mnb.predict(X_test)
gnb_predictions = gnb.predict(X_test)

# Calculate metrics for MultinomialNB
print("MultinomialNB Classification Report:")
print(classification_report(y_test, mnb_predictions))

# Calculate metrics for GaussianNB
print("GaussianNB Classification Report:")
print(classification_report(y_test, gnb_predictions))

# Step 8: Make Predictions
# TODO: Use the trained classifiers to make predictions on new data