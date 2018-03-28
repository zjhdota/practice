# -*- encoding=utf-8 -*-
import numpy as np
from keras.layers import Activation, Dense, SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist


INPUT_SIZE = 28 # 图片列 28
TIME_STEPS = 28 # 图片行 28
BATCH_INDEX = 0
BATCH_SIZE = 50
OUTPUT_SIZE = 10
CELL_SIZE = 50


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28) / 255
X_test = X_test.reshape(-1, 28, 28) / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(SimpleRNN(batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
                    units=CELL_SIZE,
                    ))

model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))



# optimizer
adam = Adam(0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# training
for i in range(4001):

    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)

    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if i % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('cost=', cost, 'accuracy=', accuracy)
