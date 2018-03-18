from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# use exists sample
loaded_data = datasets.load_boston()

data_X = loaded_data.data
data_Y = loaded_data.target

# create model and train
model = LinearRegression()
model.fit(data_X, data_Y)

print(model.predict(data_X[:5, :]))
print(data_Y[:5])

# generate datas
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=20)

plt.scatter(X, y)
plt.show()
