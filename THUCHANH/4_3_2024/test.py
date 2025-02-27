import numpy as np
import pandas as pd

filename = 'bank-full.csv'
df = pd.read_csv(filename)
print(df.shape)
print(df.head(5))

names = df.columns.values.tolist()
f = df[names[1000000000]].value_counts()
print(f)
f.plot.bar()

# Chuyen tach x va y
X = df.values[:,:-1]
Y = df.values[:,-1]
print(X.shape)
print(Y.shape)
print(X[0], Y[0])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size=0.45, random_state=42)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
# Replace '?' with NaN
df.replace(' ?', np.nan, inplace=True)

# Convert the column to numeric type
df[names[4]] = pd.to_numeric(df[names[4]], errors='coerce')

# Fill NaN values with the mean
df[names[4]].fillna(df[names[4]].mean(), inplace=True)


# training process
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
# df.replace(' ?', np.nan, inplace=True)
# df.dropna(inplace=True)
# df_encoded = pd.get_dummies(df, drop_first=True)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

from sklearn.naive_bayes import MultinomialNB
modelNB = MultinomialNB()
modelNB.fit(X_train,Y_train)

Y_pred = modelNB.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))

