'''
Function: to Extract your Weights/bias and save them as <.npy> files

Friendly Notes:

By this file, you can:
    1. Read the weights/bias from trained <.h5> file of neural network model
    2. Save as <.npy> file to specified path
    3. In the next file, we will use Inference Functions to make forward computation

'''

import numpy as np
from keras.models import load_model


model = load_model(r'C:\Users\Yunqing\Desktop\Fruit.h5')   # Load your trained model, which was saved as <.h5> file
print(model.summary())   # Show the network structure

weight_origin_1 = model.layers[1].get_weights()[0]   # Load the corresponding parameters from different Layers
weight_origin_11 = model.layers[1].get_weights()[1]  # [0] means <weights>, [1] means <bias>

'''
* Important notes: *
 
One may only read and save weights/bias of layers that do have parameters
In this network model, we only read and save parameters in layer[1], layer[3], layer[5], layer[7]

'''
weight_origin_3 = model.layers[3].get_weights()[0]
weight_origin_31 = model.layers[3].get_weights()[1]

weight_origin_5 = model.layers[5].get_weights()[0]
weight_origin_51 = model.layers[5].get_weights()[1]

weight_origin_7 = model.layers[7].get_weights()[0]
weight_origin_71 = model.layers[7].get_weights()[1]

print('cnn1_weights:', weight_origin_1.shape)  # show the shape of the matrix
print('cnn1_bias:', weight_origin_11.shape)
print('cnn2_weights:', weight_origin_3.shape)
print('cnn2_bias:', weight_origin_31.shape)

print('fc1_weights:', weight_origin_5.shape)
print('fc1_bias:', weight_origin_51.shape)

print('fc1_weights:', weight_origin_7.shape)
print('fc1_bias:', weight_origin_71.shape)

np.save('cnn_weights_1.npy', weight_origin_1)   # save the weights/bias to <.npy> file in current file path
np.save('cnn_bias_1.npy', weight_origin_11)
np.save('cnn_weights_2.npy', weight_origin_3)
np.save('cnn_bias_2.npy', weight_origin_31)

np.save('fc_weights_1.npy', weight_origin_5)
np.save('fc_bias_1.npy', weight_origin_51)
np.save('fc_weights_2.npy', weight_origin_7)
np.save('fc_bias_2.npy', weight_origin_71)

