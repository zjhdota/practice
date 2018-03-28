import numpy as np
# 按顺序建立神经网络 Sequential
from keras.models import Sequential
# 全连接层 Dense model
from keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(1337)

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + np.random.normal(0, 0.05, (200, ))

# plt.scatter(X, Y, )
# plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')

# train
print('staring train...')
for i in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if i % 50 == 0:
        print("cost=", cost)
print('finishing train...')

# test
print('starting test...')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('cost=', cost)
W, b = model.layers[0].get_weights()
print('W=', W, 'b', b)

Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test,Y_pred)
plt.show()











