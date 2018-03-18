from sklearn import datasets
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


#  加载数据集
loaded_digits = datasets.load_digits()
X = loaded_digits.data
y = loaded_digits.target

# 建立参数集
param_range = np.logspace(-6, -2.3, 5)

train_loss, test_loss = validation_curve(SVC(), X, y, param_name="gamma", param_range=param_range, cv=10, scoring='neg_mean_squared_error')

print(train_loss)

train_loss_mean = -np.mean(train_loss, axis=1)`
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color='g', label="Traing_loss_mean")
plt.plot(param_range, test_loss_mean, 'o-', color='r', label='Test_loss_mean')

plt.xlabel('Training example')
plt.ylabel('loss')
plt.legend(loc="best")
plt.show()
