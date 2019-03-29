'''
---------------------------------------------------------------------------
The Inference Computation Function:

An imitation of model prediction process in TensorFlow/Keras,
trying to solve the problem of unsuccessfully installation of TensorFlow
---------------------------------------------------------------------------
Whole steps:
1. Read the Weights from stored <.npy> files of different layer
2. Start from input tensor, make some data augmentation
3. Calculate the value layer by layer
4. Output the Result
5. In demo.py, the Result from Inference Engine will be as output to WeChat
---------------------------------------------------------------------------
'''

import numpy as np
from PIL import Image, ImageOps


class Recognizer():
    def __init__(self):
        self.conv_weights1 = np.load(r'C:\Users\Yunqing\Desktop\cnn_weights_1.npy')
        self.conv_bias1 = np.load(r'C:\Users\Yunqing\Desktop\cnn_bias_1.npy')
        self.conv_weights2 = np.load(r'C:\Users\Yunqing\Desktop\cnn_weights_2.npy')
        self.conv_bias2 = np.load(r'C:\Users\Yunqing\Desktop\cnn_bias_2.npy')
        self.fc_weights1 = np.load(r'C:\Users\Yunqing\Desktop\fc_weights_1.npy')
        self.fc_bias1 = np.load(r'C:\Users\Yunqing\Desktop\fc_bias_1.npy')
        self.fc_weights2 = np.load(r'C:\Users\Yunqing\Desktop\fc_weights_2.npy')
        self.fc_bias2 = np.load(r'C:\Users\Yunqing\Desktop\fc_bias_2.npy')
        
    def predict(self, x, img_enhance=False):
        x = x / 255.0
        if img_enhance:
            x = self.enhancement(x)
        output = self.Inference(x)
        label = int(np.where(output == np.max(output))[1])

        return output, label
        
    def Inference(self, x):
        x = self.Flatten(x)

        x = self.Dense(x, self.conv_weights1, self.conv_bias1)
        x = self.Relu(x)

        x = self.Dense(x, self.conv_weights2, self.conv_bias2)
        x = self.Relu(x)

        x = self.Dense(x, self.fc_weights1, self.fc_bias1)
        x = self.Relu(x)

        x = self.Dense(x, self.fc_weights2, self.fc_bias2)
        x = self.Normalization(x)
        return x
    
    def Relu(self, x):
        x[x < 0] = 0
        return x
        
    def Flatten(self, x):
        # Resize input Tensor to (1,n) shape
        x = x.reshape(1,-1)
        return x
    
    def Normalization(self, x):
        max_x = np.max(x)
        min_x = np.min(x)
        y = (x - min_x) / (max_x - min_x)
        return y
    
    def Conv2D(self, inputs, kernels, biases, strides=(1,1)):
        # input Tensor dimension as [heigh,width,channel]
        input_size = inputs.shape
        # kernel size dimension as [heigh,width,channel,k]
        kernel_size = kernels.shape
        # padding the input tensor according to the kernel size
        h_pad = int(kernel_size[0] / 2)
        w_pad = int(kernel_size[1] / 2)
        inputs = np.pad(inputs, pad_width=((h_pad,h_pad),(w_pad,w_pad),(0,0)), mode='constant')
        # compute the dimension of output tensor
        w_new = int((input_size[0] - kernel_size[0] + 2*h_pad) / strides[0] + 1)
        h_new = int((input_size[1] - kernel_size[1] + 2*w_pad) / strides[1] + 1)
        # initialize the output tensor with calculated dimension
        outputs = np.zeros((w_new,h_new,kernels.shape[3]))
        # convolution with channel k 
        for k in range(kernel_size[3]): # (i_new,j_new)represents the place this step in the output tensor
            #  (i,j)represents the window place in this step, by the center of (i,j)
            for i_new,i in enumerate(range(h_pad, inputs.shape[0]-h_pad, strides[0])):
                for j_new, j in enumerate(range(w_pad, inputs.shape[1]-w_pad, strides[1])):
                    # temp represents the computed value in the temporary window.
                    temp = inputs[i-h_pad:i+h_pad+1, j-w_pad:j+w_pad+1, :]
                    temp = self.Flatten(temp)
                    # kernel represents the k th kernel 
                    kernel = self.Flatten(kernels[:,:,:,k])
                    # compute the temporary value, then output to the tensor
                    # compute the temporary value, then output to the tensor
                    value = np.dot(temp, kernel.T) + biases[k]
                    outputs[i_new,j_new, k] = value

        return outputs
    
    def Max_pooling(self, x, kernel_size=(2,2), strides=(2,2)):
        # compute the dimension of output tensor
        width, heigh, channel = x.shape
        # initialize the output tensor depending onkernel_size and strides
        w_new = int((width - kernel_size[0]) / strides[0] + 1)
        h_new = int((heigh - kernel_size[1]) / strides[1] + 1)
        c_new = channel
        outputs = np.zeros((w_new,h_new, c_new))
        # pooling by layers, depends on the channel k
        for k in range(channel):
            # (i_new,j_new) as coordinates in output tensor
            for i_new,i in enumerate(range(0, width, strides[0])):
                for j_new,j in enumerate(range(0, heigh, strides[1])):
                    # (i,j)as coordinates of input tensor
                    temp = x[i:i+kernel_size[0],j:j+kernel_size[1]]
                    # (i,j)as upper left, extract the local largest value
                    outputs[i_new, j_new,k] = np.max(temp)

        return outputs
    
    def Dense(self, inputs, kernel, bias):  
        # Initialize the dimension of input tensor and bias as (1,n)
        inputs = inputs.reshape(1,-1)
        bias = bias.reshape(1,-1)
        # vectorize the output tensor
        output = np.dot(inputs, kernel) + bias

        return output
    
    def enhancement(self, x):
        mean = np.mean(x)
        x[x>=mean] = 1.0
        x[x<mean] = 0.0
        return x


model = Recognizer()
if __name__ == '__main__':
    # Use a photo to test the result
    x = (Image.open(r'C:\Users\Yunqing\Desktop\F\Fruit_Data\Fruit_data\val\xj\IMG_1015.jpg'))
    x = x.resize((50, 50))
    x = np.array(x)
    y = model.predict(x)
    print(y)
