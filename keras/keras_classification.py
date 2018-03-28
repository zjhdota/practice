import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# X shape (60000, 28 * 28) y shape(10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255 # normalize 0 ~ 1
X_test = X_test.reshape(X_test.shape[0], -1) / 255 # normalize 0 ~ 1
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# model
model = Sequential([
                   Dense(units=32, input_dim=784),
                   Activation('relu'),
                   Dense(units=10),
                   Activation('softmax')
    ])

# optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

# train
model.fit(X_train, y_train, epochs=2, batch_size=32)

# test
loss, accuracy = model.evaluate(X_test, y_test)

print('loss=', loss)
print('accuracy=', accuracy)




























