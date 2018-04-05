import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# sns.set(style='white', color_codes=True)
# iris = sns.load_dataset('iris')

# iris.plot(kind='scatter', x='sepal_length', y='sepal_width')
# sns.pairplot(iris, hue='species')
# plt.show()

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()


# k折交叉验证
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = cross_val_score(model, X_test, y_pred, scoring='accuracy', cv=10).mean()
print(score)


# 留出法
loo = LeaveOneOut()
accuracy = 0.0
for train, test in loo.split(X):
    model.fit(X[train], y[train])
    y_pred = model.predict(X[test])
    if y_pred == y[test]:
        accuracy += 1

print(accuracy / np.shape(X)[0])


