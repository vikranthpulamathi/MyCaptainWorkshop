import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("MNIST_CSV/mnist_test.csv")

# view column heads
print(data.head())

# extracting data from the dataset and viewing them up close
a = data.iloc[4, 1:].values

# reshape the extracted data
a = a.reshape(28, 28).astype('uint8')
#print(a)

plt.imshow(a)
plt.show()

df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

# creating test and train batches
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

# check
print(x_train.head())
print(y_train.head())

# call RF classifier
rf = RandomForestClassifier(n_estimators=100)

# fit the model
rf.fit(x_train, y_train)

# prediction on test data
pred = rf.predict(x_test)
print(pred)

# check prediction accuracy
s = y_test.values

# calculate number of correctly predicted values
count = 0
for i in range(len(pred)):
  if pred[i] == s[i]:
    count += 1

print("Count of correct predictions = ", count)
print("Total vlues of prediction = ", len(pred))
print("Accuracy = ", count/len(pred)*100, "%")
