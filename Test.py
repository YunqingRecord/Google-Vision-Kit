import keras
from keras import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
import time
start = time.time()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)  # (data_size, width, height,channel)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)  # (data_size, width, height,channel)

M_class = 10         # number of class ranging (0~9)
M_epoch = 200
M_batch_size = 128   # 2^7

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.   # scaling the data, divide them by 255, since value of pixel ranges in (0, 255)
x_test /= 255.

num_kernel = 32  # num of feature to scratch

kernel_size = (3, 3)  # to define the compute size


y_train = to_categorical(y_train, M_class)  # One Hot the label of the Mnist Images
y_test = to_categorical(y_test, M_class)

model = Sequential()   # Another kind of Model is Functional, use Sequential here

model.add(Conv2D(num_kernel, kernel_size[0], kernel_size[1], input_shape=(28, 28, 1)))
model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=2, strides=1))  # average intensity in 4*4 region
model.add(Conv2D(num_kernel, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))


model.add(Flatten())   # transfer 7*7 into 49, one column pixel

# First Hidden Layer
model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dropout(0.2))

# Second Hidden Layer
model.add(Dense(48))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Third Hidden Layer
model.add(Dense(16))
model.add(Activation('relu'))

# Output Layer
model.add(Dense(10))
model.add(Activation('softmax'))

# using sgd optimizer
sgd = SGD(lr=0.5, decay=1e-6, momentum=0.95, nesterov=True)
# sgd = SGD(lr=0.5)

model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])

# call back the performance using history.history
history = model.fit(x_train, y_train, validation_split=0.2, batch_size=M_batch_size, epochs=M_epoch, shuffle=True)
model.evaluate(x_test, y_test, batch_size=128)
loss, accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)

model.summary()

# Plot the loss of the model
plt.figure(1)
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'b')
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss", "val_loss"])
plt.savefig('C:\\Users\Yunqing\Desktop\SEM2\/fuzzy\Assignment\cnnLoss.png')

# Plot the acuuracy of the model
plt.figure(2)
plt.plot(history.history['acc'], 'g')
plt.plot(history.history['val_acc'], 'y')
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train_acc", "val_acc"], loc="upper right")
plt.savefig('C:\\Users\Yunqing\Desktop\SEM2\/fuzzy\Assignment\cnnAccuracy.png')

plt.show()

plot_model(model, to_file='C:\\Users\Yunqing\Desktop\SEM2\/fuzzy\Assignment\MnistCNN.png', show_shapes='True')

model.save('CNN_MNIST.h5')

end = time.time()
print("Total Time used by Keras Framework :", end-start)
