import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = np.loadtxt('watermelon_data.csv', delimiter=',')

X = data[:, 1:3]
y = data[:, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)

# sover=['lsqr', 'svd', 'eigen'] 最小二乘,奇异值分解,特征分解
model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
