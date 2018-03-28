import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

# X shape (60000, 28 * 28) y shape(10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
X_train = X_train.reshape(-1, 28, 28, 1) / 255
X_test = X_test.reshape(-1, 28, 28, 1) / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# model
model = Sequential()

# layer 1 output (32, 28, 28)
model.add(
        Conv2D(filters=32, # 特征数量, 卷积核数目
               kernel_size=(5,5), # patch大小 5 * 5
               padding='same', # padding method
               input_shape=(28, 28, 1))
    )
model.add(Activation('relu'))

# pooling output shape(32, 14, 14)
model.add(MaxPooling2D(
                       pool_size=(2, 2), # 池化核大小
                       strides=(2, 2), # 池化核步长 2
                       padding='same', # padding
                        )
    )

# layer 2
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))

# pooling output size (64, 7, 7)
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

# fully connected layer 1 Flatten 降维为 1 维
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# fully connected layer 2
model.add(Dense(10))
model.add(Activation('softmax'))

# optimizer
adam = Adam(lr=1e-4)

model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])

# train
model.fit(X_train, y_train, epochs=1, batch_size=32)

# test
loss, accuracy = model.evaluate(X_test, y_test)

print('loss=', loss)
print('accuracy=', accuracy)



