import numpy as np
import math
import pandas as pd
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import csv
class Layer:
    def __init__(self):
        self.X = None
        self.Y = None
    def forward(self):
        pass
    def backward(self):
        pass
###########################        Dense Layer                  ############################
class Dense(Layer):
    def __init__(self,  out_shape): # i = input size, j= output size
        self.out_shape = out_shape
        self.weight = None
        self.bias = None

        # self.weight = np.random.randn(j,i)
        # self.bias = np.random.randn(j,1)

    def forward(self, X):
        self.X = X
        
        if self.weight is None:
            #  var = math.sqrt(2 / X.shape[0])
            #  self.weight = np.random.randn(self.out_shape, X.shape[0])* var
             self.weight = np.random.uniform(-np.sqrt(6 / (X.shape[0] + X.shape[1])), np.sqrt(6 / (X.shape[0] + X.shape[1])), (self.out_shape, X.shape[0]))
        if self.bias is None:
            self.bias = np.zeros((self.out_shape, 1))
        Y = np.dot(self.weight,self.X) + self.bias ##### Error hole etar multiply dekh
        # print(Y.shape)
        return Y
    
    def backward(self,Y_gradient, learning_rate):
        # print(Y_gradient.shape)
        weight_gradient = np.dot(Y_gradient, self.X.T)
        # print(self.weight.shape)
        # print(Y_gradient.shape)
        X_gradient = np.dot(self.weight.T, Y_gradient)

        self.weight -=learning_rate * weight_gradient
        self.bias -= learning_rate * Y_gradient

        return X_gradient
###########################        Convolution Layer            ############################
class Convolution(Layer):
    def __init__(self, output_channel, filter_size, input_shape=None, stride=1, padding=0):
        # output channel : number of kernels
        # input shape : image depth * height * width
        # filter size : 2d shape of kernel (d x d)

        self.input_channel = 0
        self.output_height = 0
        self.output_width = 0
        self.output_channel = output_channel
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        

        self.kernel = None
        self.bias = None   
    def pad(self, input, padding):
        return np.pad(input, [(0,0), (padding, padding), (padding, padding)], mode='constant')
    
    def changeGradient(self, gradient, stride = None, padding=True):
        new_gradient = gradient
        if self.stride != 1:
            new_gradient = np.insert(new_gradient, np.repeat(range(1, new_gradient.shape[1]),self.stride-1), 0, axis = 1)
            new_gradient = np.insert(new_gradient, np.repeat(range(1, new_gradient.shape[2]),self.stride-1), 0, axis = 2)

        if padding:
            new_gradient = self.pad(new_gradient, self.filter_size[0]-1)

        return new_gradient

    def conv(self, input, kernel, stride=1, padding=0):
        if padding != 0:
            input = self.pad(input,padding)
        height = math.floor((input.shape[0]-kernel.shape[0])/self.stride)+1
        width = math.floor((input.shape[1]-kernel.shape[1])/self.stride)+1
        output = np.zeros((height,width))
        for h in range(0, height):
            i = h*self.stride
            for w in range(0, width):
                j = w*self.stride
                output[h][w] = np.sum(input[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
        
        return output

    def forward(self, input):
        if self.kernel is None:
            input_shape = input.shape
            depth, height, width = input_shape
            self.input_channel = depth
            self.output_height = math.floor((height + 2*self.padding - self.filter_size[0]) / self.stride) + 1
            self.output_width = math.floor((width + 2*self.padding - self.filter_size[1]) / self.stride) + 1

            self.kernel = np.random.uniform(-np.sqrt(6 / (input.shape[1] + input.shape[2])), np.sqrt(6 / (input.shape[1] + input.shape[2])),(self.output_channel,depth,self.filter_size[0], self.filter_size[1]))
            self.bias = np.random.randn(self.output_channel,self.output_height, self.output_width)
        
        # main forward propagation
        self.input = input
        tmp_input = self.pad(input, self.padding)
        self.output = self.bias
        for i in range(self.output_channel):
            for j in range(tmp_input.shape[0]):
                self.output[i] += self.conv(tmp_input[j],self.kernel[i][j])
        return self.output
    
    def backward(self, gradient, learning_rate):

        wrt_kernel = np.zeros(self.kernel.shape)
        wrt_input = np.zeros(self.input.shape)

        kernel_gradient = self.changeGradient(gradient,padding=False)
        input_gradient = self.changeGradient(gradient)

        for i in range(self.output_channel):
            for j in range(self.input.shape[0]):
                wrt_kernel[i,j] = self.conv(self.input[j],kernel_gradient[i])
                wrt_input[j] += self.conv(input_gradient[i], np.rot90(self.kernel[i][j], 2))

        self.kernel -= learning_rate * wrt_kernel
        self.bias -= learning_rate * gradient

        return wrt_input
###########################        MaxPooling                   ############################
class MaxPooling(Layer):
    def __init__(self, filter_size, stride=1, input_shape=None):
        self.filter_size = filter_size
        self.output_shape = None
        self.output = None
        self.stride = stride
        if input_shape is not None:
            depth, height, width = input_shape
            out_height = math.floor((height - filter_size[0]) / stride) + 1
            out_width = math.floor((width - filter_size[1]) / stride )+ 1

            self.output_shape = (depth, out_height, out_width)
            self.output = np.zeros(self.output_shape)

    def forward(self, input):
        if self.output is None:
            depth, height, width = input.shape
            out_height =math.floor( (height - self.filter_size[0]) / self.stride) + 1
            out_width = math.floor((width - self.filter_size[1]) / self.stride) + 1

            self.output_shape = (depth, out_height, out_width)
            self.output = np.zeros(self.output_shape)
        
        depth, height, width = self.output_shape
        self.input = input
        for ch in range(depth):
            for i in range(height):
                ii = i*self.stride
                for j in range(width):
                    jj = j*self.stride
                    self.output[ch,i,j] = np.max(self.input[ch, ii:ii+self.filter_size[0], jj:jj+self.filter_size[1]])
        return self.output
    
    def backward(self, gradient, learning_rate):
        depth, height, width = self.output_shape
        wrt_input = np.zeros(self.input.shape)

        for ch in range(depth):
            for i in range(height):
                ii = i*self.stride
                for j in range(width):
                    jj = j*self.stride
                    val = np.max(self.input[ch, ii:ii+self.filter_size[0], jj:jj+self.filter_size[1]])
                    maxi, maxj = np.where(val == self.input[ch, ii:ii+self.filter_size[0], jj:jj+self.filter_size[1]])
                    wrt_input[ch, ii:ii+self.filter_size[0], jj:jj+self.filter_size[1]][maxi[0]][maxj[0]] = gradient[ch,i,j]
        return wrt_input
###########################        Flattening                   ############################
class Flattening(Layer):
    def __init__(self, input_shape=None, output_shape=None):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input_shape = input.shape

        self.output_shape = (input.size,1)
        self.input = input
        self.output = np.reshape(input, self.output_shape)

        return self.output
    
    def backward(self, gradient, learning_rate):
        reshaped = np.reshape(gradient, self.input_shape)
        return reshaped

###########################        ReLU Activation Layer        ############################
def relu(x):
    a = np.maximum(0,x)
    return a
def relu_prime(x):
    a = x > 0 ##########################
    b = a.astype(int) 
    # print(b)
    return b

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, X):
        self.X = X
        self.Y = self.activation(self.X)
        return self.Y

    def backward(self, Y_gradient, learning_rate):
        a = np.multiply(Y_gradient, self.activation_prime(self.X))
        return a
###########################        Cross entropy                 ############################
def cross_entropy(y, y_pred):
    m = y.shape[0]
    epsilon = 1e-5
    loss = -(1 / m) * np.sum(y * np.log(y_pred+epsilon))
    return loss

def cross_entropy_prime(y, y_pred):
    m = y.shape[0]
    # print("cross y: ",y.shape)
    # print("cross y_pred: ",y_pred.shape)
    derivative = (1 / m) * (y_pred - y)
    # print("cross: ",derivative.shape)
    return derivative

def mean_square_error(y, y_pred):
    a = (y_pred - y) ** 2
    sum = np.sum(a)
    m = y.shape[0]
    return sum/m

def mean_square_error_prime(y, y_pred):
    m = y.shape[0]
    a = (2*(y-y_pred)) / m
    return a
###########################        Softmax                      ############################
class Softmax(Layer):
    def forward(self, X):
        self.X = X
        m = np.max(X)
        a = np.exp(self.X - m)
        b = np.sum(a)
        self.Y = a/b
        # print("into soft forward:", self.Y.shape)
        return self.Y

    def backward(self, Y_gradient, learning_rate):
        a = self.Y *(np.identity(self.Y.shape[0]) - self.Y)
        # print("into soft backward:",Y_gradient.shape)
        output= np.matmul(a, Y_gradient)
        
        return Y_gradient



###########################        Model                        ############################
class Model():
        def __init__(self, learning_rate, epochs, error, error_prime):
            self.learning_rate= learning_rate
            self.epochs = epochs
            self.network = []
            self.error = error
            self.error_prime = error_prime
        def add(self, component):
            self.network.append(component)
        def train(self, X,Y):
            for epoch in range(self.epochs):
                err = 0
                for x,y in zip(X,Y):
                    y_pred = x
                    for net in self.network:
                        y_pred = net.forward(y_pred)
                    # print(y.T,y_pred)
                    err += self.error(y.T, y_pred)
                    # print(err)

                    derivative= self.error_prime(y.T,y_pred)
                    var = reversed(self.network)
                    for net in var:
                        derivative = net.backward(derivative, self.learning_rate)
                err /= len(X)
                # if epoch % 100 == 0:
                print('epoch',epoch+1,':',err)
        def predict(self, X):
            y_pred = X
            for net in self.network: # train korar por prediction. train korle network ta learn kore. then akta x dile ota oi learn onujayi predit kore
                y_pred = net.forward(y_pred)
            return y_pred
def run(X,Y):
    epochs = 10
    lr = 0.01
    model = Model(learning_rate=lr, epochs=epochs, error = cross_entropy, error_prime = cross_entropy_prime)
    # model.add(Convolution(20, (5,5)))
    # model.add(Activation(relu,relu_prime))
    # model.add(Convolution(12, (3,3)))
    # model.add(Activation(relu,relu_prime))
    # model.add(MaxPooling((2,2)))

    # model.add(Flattening())
    # model.add(Dense(30))
    # model.add(Activation(relu,relu_prime))
    # model.add(Dense(10))
    # model.add(Softmax())

    ###### LeNet
    model.add(Convolution(6, (5,5)))
    model.add(Activation(relu,relu_prime))
    model.add(MaxPooling((2,2)))
    model.add(Convolution(16, (5,5)))
    model.add(Activation(relu,relu_prime))
    model.add(MaxPooling((2,2)))
    model.add(Flattening())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(10))
    model.add(Softmax())
    

    model.train(X,Y)
    return model

def load_data(filename, dim_size = None):
    data = pd.read_csv(filename)
    data = data.loc[:, ['filename', 'digit',]].values
    folder = filename.split('.')[0]
    x = []
    y = []
    for name in data[:,0]:
        low_folder = folder
        img_dir = os.path.join(low_folder,name)
        img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if dim_size is not None:
            img=cv2.resize(img,(dim_size,dim_size),interpolation=cv2.INTER_AREA)
            img = np.reshape(img, (1,dim_size,dim_size))
            img = (img-np.mean(img)) / np.std(img)
        x.append(img)
    x = np.array(x)
    y_label = data[:,1]
    y = np.zeros((y_label.shape[0],1,10))
    for i in range(y_label.shape[0]):
        y[i,0,y_label[i]] = 1 
    return (x,y)

def testFileLoader(folder_dir, dim_size = None):
    x=[]
    image = []
    for images in os.listdir(folder_dir):
        if images.endswith(".png") or images.endswith(".jpg")  or images.endswith(".jpeg") :
            img = cv2.imread(os.path.join(folder_dir,images), cv2.IMREAD_COLOR)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            image.append(images)

            if dim_size is not None:
                img=cv2.resize(img,(dim_size,dim_size),interpolation=cv2.INTER_AREA)
                img = np.reshape(img, (1,dim_size,dim_size))
                img = (img-np.mean(img)) / np.std(img)

            # print(type(x))
            x.append(img)
    x = np.array(x)
    return x,image                  


folder_dir = r"test-a1"
dim_size = 28
X, images = testFileLoader(folder_dir, dim_size)
model = pickle.load(open('1705028_model.pkl', 'rb'))
x_test = X[0:2727,:,:,:]
# x_test = X[0:100,:,:,:]
y_pred = []
for x in x_test:
    y_pred.append(model.predict(x).T)
vals = np.argmax(y_pred,axis=2)
print(vals)
with open('1705028_predicted.csv', 'w', newline='') as f:

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    print(len(images))
    for i in range(len(vals)):
        # print(images[i], vals[i])
        row = [str(images[i]),str(vals[i,0])]
        writer.writerow(row)