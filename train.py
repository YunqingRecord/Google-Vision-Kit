'''
The whole steps of Google Vision Kit Recognition task are:
--------------------------------------------------------------------------------------------
1. Using Keras(tensorflow as backend) to train your neural network model
2. Apply functions of <model.save()> to save your trained model as <Your_File_Name.h5> file
3. Load your model and get weights/bias from different layers, which have parameters
4. Transfer your weights/bias into <YourWeights.npy> by using <np.save()>
5. By <np.load()> function, you can load your network weights/bias into variables
6. Compute the inference/prediction value by module "Inference Engine" forwardly
7. Once completed the computation process, output the prediction value to WeChat Room
8. Step 1-7 is one iteration, continue to make prediction by pressing shoot button of Kit
--------------------------------------------------------------------------------------------
This is for training your own Image Datasets

The weights of trained model will be uploaded to the Google Vision Kit by WinSCP

Make inference forwardly image by image,

and then return the predicted value to WeChat Chatting room
'''

from keras import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
import time
start = time.time()

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Load MNIST data

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)  # (data_size, width, height,channel)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)  # (data_size, width, height,channel)

M_class = 10         # number of class ranging (0~9)
M_epoch = 50
M_batch_size = 128   # 2^7

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.   # scaling the data, divide them by 255, since value of pixel ranges in (0, 255)
x_test /= 255.

y_train = to_categorical(y_train, M_class)  # One-Hot the label of the Mnist Images
y_test = to_categorical(y_test, M_class)

# Very shallow CNN, taken limited computational resource into consideration
model = Sequential()   # Another kind of Model is Functional, use Sequential here
model.add(Conv2D(input_shape=(28, 28, 1), filters=4, kernel_size=(3, 3), strides=(1, 1), padding='SAME'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='SAME'))  # average intensity in 4*4 region
model.add(Activation('relu'))

model.add(Flatten())


# # # First Hidden Layer
model.add(Dense(32))
model.add(Activation('relu'))


# Output Layer for ten classification
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

model.summary()   # PLot the network Structure

# Plot the loss of the model
plt.figure(1)
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'b')
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss", "val_loss"])

# Plot the acuuracy of the model
plt.figure(2)
plt.plot(history.history['acc'], 'g')
plt.plot(history.history['val_acc'], 'y')
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train_acc", "val_acc"], loc="upper right")

plt.show()

model.save('Cnn_Mnist.h5')  # Save model as <YourName.h5> make it easy to get weights.

