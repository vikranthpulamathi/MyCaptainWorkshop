import sys
import scipy
import matplotlib
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn import datasets

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# Load dataset
dataset = pd.read_csv('IRIS.csv', names=names)

# Convert feature columns to numeric (in case they are read as strings)
for col in names[:-1]:  # exclude the 'class' column
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

# Check for non-numeric entries converted to NaN
print(dataset.isnull().sum())

# Drop rows with NaN (optional, only if needed)
dataset.dropna(inplace=True)

# Print class distribution
print(dataset.groupby('class').size())

#print(dataset.shape)
#print(dataset.head(20))

#print(dataset.describe())

# Class Distribution
print(dataset.groupby('class').size())

# Univariate Plots - Box and Whisker
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.suptitle('Box and Whisker Plots of Iris Features')
pyplot.show()

dataset.hist()
pyplot.show()

# Multivariate Plots
scatter_matrix(dataset)
pyplot.show()

# Validation Dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=1)

# LogisticRegression
# LinearDiscriminantAnalysis
# K-Nearest Neighbors
# Classification and Regression Trees
# Gaussian Naive Bayes
# Support Vector Machines

# Building Models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

for name, model in models:
  kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare models
pyplot.boxplot(results, tick_labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on SVM
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

