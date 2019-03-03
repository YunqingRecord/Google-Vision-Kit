
# coding: utf-8

# ## 网络基本组件

import numpy as np

# In[144]:


class Recognizer():
    def __init__(self):
        self.conv_weights = np.load('model/weights/conv_weights.npy')
        self.conv_bias = np.load('model/weights/conv_bias.npy')
        self.fc1_weights = np.load('model/weights/fc1_weights.npy')
        self.fc1_bias = np.load('model/weights/fc1_bias.npy')
        self.fc2_weights = np.load('model/weights/fc2_weights.npy')
        self.fc2_bias = np.load('model/weights/fc2_bias.npy')
        
    def predict(self, x, img_enhance=False):
        x = x.reshape((28,28,1))
        x = x / 255.0
        if img_enhance:
            x = self.enhancement(x)
        output = self.Inference(x)
        label = int(np.where(output==np.max(output))[1])
        
        return  output,label
        
    def Inference(self,x):
        x = self.Conv2D(x, self.conv_weights, self.conv_bias, strides=(1,1))
        x = self.Relu(x)
        x = self.Max_pooling(x, kernel_size=(2,2), strides=(2,2))
        x = self.Flatten(x)
        x = self.Dense(x, self.fc1_weights, self.fc1_bias)
        x = self.Relu(x)
        x = self.Dense(x, self.fc2_weights, self.fc2_bias) 
        x = self.Normalization(x)
        return x
    
    def Relu(self, x):
        x[x<0] = 0
        return x
        
    def Flatten(self, x):
        # 将输入张量resize到(1,n)形式
        x = x.reshape(1,-1)
        return x
    
    def Normalization(self, x):
        max_x = np.max(x)
        min_x = np.min(x)
        y = (x - min_x) / (max_x - min_x)
        return y
    
    def Conv2D(self, inputs, kernels, biases, strides=(1,1)):
        # 计算输入张量的维度以[heigh,width,channel], 卷积核的维度以[heigh,width,channel,k]
        input_size = inputs.shape
        kernel_size = kernels.shape
        # 对输入张量按照卷积核的尺寸进行pading
        h_pad = int(kernel_size[0] / 2)
        w_pad = int(kernel_size[1] / 2)
        inputs = np.pad(inputs, pad_width=((h_pad,h_pad),(w_pad,w_pad),(0,0)), mode='constant')
        # 计算输出张量的维度，并初始化输出张量
        w_new = int((input_size[0] - kernel_size[0] + 2*h_pad) / strides[0] + 1)
        h_new = int((input_size[1] - kernel_size[1] + 2*w_pad) / strides[1] + 1)
        outputs = np.zeros((w_new,h_new,kernels.shape[3]))
        # 按通道k逐层卷积
        for k in range(kernel_size[3]):
            # (i_new,j_new)代表该步在输出张量中对应的位置，(i,j)代表该步中以输入张量(i,j)为中心的窗口位置
            for i_new,i in enumerate(range(h_pad, inputs.shape[0]-h_pad, strides[0])):
                for j_new,j in enumerate(range(w_pad, inputs.shape[1]-w_pad, strides[1])):
                    # temp代表当前位置的滑动窗口所取的值
                    temp = inputs[i-h_pad:i+h_pad+1, j-w_pad:j+w_pad+1, :]
                    temp = self.Flatten(temp)
                    # kernel代表当前第k个卷积核
                    kernel = self.Flatten(kernels[:,:,:,k])
                    # 计算当前窗口的卷积值，并赋值到输出张量中
                    value = np.dot(temp, kernel.T) + biases[k]
                    outputs[i_new,j_new, k] = value

        return outputs
    
    def Max_pooling(self, x, kernel_size=(2,2), strides=(2,2)):
        # 计算输出张量的尺寸
        width, heigh, channel = x.shape
        # 按照kernel_size和strides创建输出张量
        w_new = int((width - kernel_size[0]) / strides[0] + 1)
        h_new = int((heigh - kernel_size[1]) / strides[1] + 1)
        c_new = channel
        outputs = np.zeros((w_new,h_new, c_new))
        # 按通道k逐层池化
        for k in range(channel):
            # (i_new,j_new)为输出张量中的坐标，(i,j)为输入张量中对应的坐标
            for i_new,i in enumerate(range(0, width, strides[0])):
                for j_new,j in enumerate(range(0, heigh, strides[1])):
                    # 以(i,j)坐标为左上角，取局部窗口内的值
                    temp = x[i:i+kernel_size[0],j:j+kernel_size[1]]
                    # 提取当前窗口内的最大值
                    outputs[i_new,j_new,k] = np.max(temp)

        return outputs
    
    def Dense(self, inputs, kernel, bias):  
        # 规范输入张量及bias的维度为(1,n)
        inputs = inputs.reshape(1,-1)
        bias = bias.reshape(1,-1)
        # 向量化形式计算当前层的输出值
        output = np.dot(inputs, kernel) + bias

        return output
    
    def enhancement(self, x):
        mean = np.mean(x)
        x[x>=mean] = 1.0
        x[x<mean] = 0.0
        return x

