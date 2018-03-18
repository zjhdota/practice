from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score # K折交叉验证模块
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 获得数据
iris = load_iris()
X = iris.data
y = iris.target

# 未交叉验证
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

"""
# 建立模型
knn = KNeighborsClassifier()

# 训练模型
knn.fit(X_train, y_train)

print("普通精确度", knn.score(X_test, y_test))

# 使用K折交叉验证模块
print("K折交叉验证后的精确度", cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean())
"""

# 准确率判断分类模型的好坏
k_range = range(1, 31)
k_score = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_score.append(scores.mean())

plt.plot(k_range, k_score)
plt.xlabel('value of k for KNN')
plt.ylabel('cross validation accuracy')
plt.show()

# 平方均差判断回归模型的好坏

k_range = range(1, 31)
k_score = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')
    k_score.append(loss.mean())

plt.plot(k_range, k_score)
plt.xlabel('value of k for KNN')
plt.ylabel('cross validation loss')
plt.show()
