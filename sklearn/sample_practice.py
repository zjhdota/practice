from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()

data_X = loaded_data.data
data_Y = loaded_data.target

model = LinearRegression()

model.fit(data_X, data_Y)

print(model.predict(data_X[:4, :]))

# 输出斜率
print("斜率=", model.coef_)

# 输出截距
print("截距=", model.intercept_)

# 输出参数
print("parmas=", model.get_params())

# R^2的方式打分
print("accuracy", model.score(data_X, data_Y))

