from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# 生成适合做classification资料的模块
from sklearn.datasets.samples_generator import make_classification
# Support Vector Machine中的Support Vector Classifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# 建立arrary
a = np.array([[10, 2.7, 3.6],
          [-100, 5, -2],
          [120, 20, 40]
          ])

# normalization
print(preprocessing.scale(a))

# 生成具有2种属性的300笔数据
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2, random_state=22, n_clusters_per_class=1, scale=100)
X = preprocessing.scale(X)



print("X=", X, "y=", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# 可视化数据
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

