# -*- coding: utf-8 -*-
"""task2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rS1ReWybtLRApKMeSEmNdqhM1eYR5Ko3

Overfitting classification
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/fruit.csv')
df.sample(10)

X = df.iloc[:,3:6].values
Y = df.iloc[:,0].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

from sklearn import tree

clf_2 = tree.DecisionTreeClassifier(min_samples_split = 2)

clf_2.fit(X_train, y_train)

pred_2 = clf_2.predict(X_test)

print(pred_2)

pred_train = clf_2.predict(X_train)
print(pred_train)

from sklearn.metrics import confusion_matrix, precision_score,recall_score , accuracy_score
print("Accuracy score on test data",accuracy_score(y_test,pred_2))
print("Accuracy score on training data",accuracy_score(y_train,pred_train))

# Initialize the decision tree classifier with minimum samples split as 10
# min_samples_split = 10: Tham số này xác định số lượng mẫu dữ liệu tối thiểu cần thiết để tiếp tục chia một nút.
# Ở đây, giá trị được đặt là 10, có nghĩa là mỗi nút cần ít nhất 2 mẫu dữ liệu để có thể chia thành các nút con.
clf_pruned = tree.DecisionTreeClassifier(min_samples_split=10)

# Train the decision tree classifier
clf_pruned.fit(X_train, y_train)

# Predictions on the test set
pred_test_pruned = clf_pruned.predict(X_test)
print("Accuracy score on test data with pruning:", accuracy_score(y_test, pred_test_pruned))

# Predictions on the training set
pred_train_pruned = clf_pruned.predict(X_train)
print("Accuracy score on training data with pruning:", accuracy_score(y_train, pred_train_pruned))

"""Overfitting regression"""

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/fruit.csv')
df.sample(10)

X = df.iloc[:, 3:6].values
Y = df.iloc[:, 0].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeRegressor

reg_2 = DecisionTreeRegressor(min_samples_split=2)

reg_2.fit(X_train, y_train)

pred_2 = reg_2.predict(X_test)

print(pred_2)

pred_train = reg_2.predict(X_train)
print(pred_train)

from sklearn.metrics import mean_squared_error, r2_score

mse_2 = mean_squared_error(y_test, pred_2)
r2_2 = r2_score(y_test, pred_2)

print("Mean Squared Error (Test Set):", mse_2)
print("R-squared Score (Test Set):", r2_2)

mse_train = mean_squared_error(y_train, pred_train)
r2_train = r2_score(y_train, pred_train)

print("Mean Squared Error (Training Set):", mse_train)
print("R-squared Score (Training Set):", r2_train)

reg_pruned = DecisionTreeRegressor(min_samples_split=10)

# Train the decision tree regressor
reg_pruned.fit(X_train, y_train)

# Predictions on the test set
pred_test_pruned = reg_pruned.predict(X_test)

# Calculate evaluation metrics for the test set
mse_test_pruned = mean_squared_error(y_test, pred_test_pruned)
r2_test_pruned = r2_score(y_test, pred_test_pruned)

print("Mean Squared Error (Test Set) with pruning:", mse_test_pruned)
print("R-squared Score (Test Set) with pruning:", r2_test_pruned)

# Predictions on the training set
pred_train_pruned = reg_pruned.predict(X_train)

# Calculate evaluation metrics for the training set
mse_train_pruned = mean_squared_error(y_train, pred_train_pruned)
r2_train_pruned = r2_score(y_train, pred_train_pruned)

print("Mean Squared Error (Training Set) with pruning:", mse_train_pruned)
print("R-squared Score (Training Set) with pruning:", r2_train_pruned)

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
pred_test = logreg.predict(X_test)
pred_proba_test = logreg.predict_proba(X_test)
print("Predictions on test data:", pred_test)
print("Predicted probabilities on test data:", pred_proba_test)