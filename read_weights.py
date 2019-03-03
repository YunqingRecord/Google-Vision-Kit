import keras
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop, Adam
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.models import load_model


model = load_model('Cnn_Mnist.h5')
print(model.summary())

weight_origin_0 = model.layers[0].get_weights()[0]
weight_origin_01 = model.layers[0].get_weights()[1]

weight_origin_5 = model.layers[4].get_weights()[0]
weight_origin_51 = model.layers[4].get_weights()[1]

weight_origin_6 = model.layers[6].get_weights()[0]
weight_origin_61 = model.layers[6].get_weights()[1]

print('cnn_weights:', weight_origin_0.shape)
print('cnn_bias:', weight_origin_01.shape)
print('fc_weights:', weight_origin_5.shape)
print('fc_bias:', weight_origin_51.shape)
print('fc_weights:', weight_origin_6.shape)
print('fc_bias:', weight_origin_61.shape)

np.save('cnn_weights_00.npy', weight_origin_0)
np.save('cnn_bias_00.npy', weight_origin_01)
np.save('fc_weights_00.npy', weight_origin_5)
np.save('fc_bias_00.npy', weight_origin_51)
np.save('fc2_weights_06.npy', weight_origin_6)
np.save('fc2_bias_06.npy', weight_origin_61)

