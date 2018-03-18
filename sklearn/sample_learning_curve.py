from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np

digist = load_digits()
X = digist.data
y = digist.target

train_sizes, train_loss, test_loss = learning_curve(SVC(gamma=10), X, y, cv=10, scoring="neg_mean_squared_error")

train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color='g', label='Training')
plt.plot(train_sizes, test_loss_mean, 'o-', color='r', label='cross validation')

plt.xlabel('Training example')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()


