import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import cv2
import pickle
from sklearn.metrics import accuracy_score,f1_score

import matplotlib.pyplot as plt


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward_prop(self, input):
        # will be implemented in iherited classes
        pass
    
    def backward_prop(self, gradient, learning_rate):
        # will be implemented in iherited classes
        pass

    def print_weight(self):
        pass
    def show(self):
        pass
def xavier_init(input_size, output_size, uniform=True):
    if uniform:
        init_range = np.sqrt(6.0 / (input_size + output_size))
        return np.random.uniform(-init_range, init_range, (input_size, output_size))
    else:
        stddev = np.sqrt(2.0 / (input_size + output_size))
        return np.random.normal(0, stddev, (input_size, output_size))
class DenseLayer(Layer):
    def __init__(self, output_shape, input_shape = None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        if input_shape is not None:
            self.weight = np.random.randn(self.output_shape, self.input_shape) * np.sqrt(2 / self.input_shape)
            # self.weight = xavier_init(self.output_shape, self.input_shape)
            self.bias = np.random.randn(self.output_shape, 1)

    def forward_prop(self, input):
        self.input = input
        if self.input_shape is None:
            # print(input.shape)
            self.input_shape = input.shape[0]
            self.weight = np.random.randn(self.output_shape, self.input_shape) * np.sqrt(2 / self.input_shape)
            # self.weight = xavier_init(self.output_shape, self.input_shape)
            self.bias = np.random.randn(self.output_shape, 1)
            # print('denseeeeeee')
            # print(self.weight)
            # print(self.bias)
        # print(input.shape)
        self.output = np.matmul(self.weight, self.input) + self.bias
        return self.output

    def backward_prop(self, gradient, learning_rate):
        wrt_weight = np.matmul(gradient, self.input.T)
        wrt_input = np.matmul(self.weight.T, gradient)
        self.weight -= learning_rate * wrt_weight 
        self.bias -= learning_rate * gradient

        return wrt_input

    def show(self):
        print('Dense Layer')
        print(self.weight)
        print(self.bias)
# activation functions

class Activation(Layer):
    def __init__(self, non_linear, non_linear_prime):
        # activation function and its first derivative
        self.non_linear = non_linear
        self.non_linear_prime = non_linear_prime

    def forward_prop(self, input):
        self.input = input
        self.output = self.non_linear(self.input)
        return self.output

    def backward_prop(self, gradient, learning_rate):
        wrt_input = gradient * self.non_linear_prime(self.input)
        return wrt_input
    def show(self):
        print('Activation')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    return np.tanh(x)
def tanh_prime(x):
    return 1 - np.tanh(x)**2

class Relu(Layer):
    def forward_prop(self, input, alpha = 0.01):
        self.input = input
        self.output = np.maximum(0,input)
        # self.output = np.maximum(alpha * input, input)
        return self.output
    def backward_prop(self, gradient, learning_rate):
        b = np.copy(self.input)
        b[b >= 0] = 1
        b[b < 0] = 0
        return gradient*b

class Softmax(Layer):
    def forward_prop(self, input):
        # print('softmax')
        # print(input)
        self.input = input
        m = np.max(input)
        exp = np.exp(input-m)
        self.output = exp / np.sum(exp)
        return self.output

    def backward_prop(self, gradient, learning_rate):
        # n = self.output.shape[0]
        # return np.matmul(self.output * (np.identity(n) - self.output), gradient)
        return gradient
    
    def show(self):
        print('Softmax')

# error/loss functions

def mse(y, y_pred):
    tmp = (y - y_pred) ** 2
    return np.mean(tmp)

def mse_prime(y, y_pred):
    n = y.shape[0]
    return 2*(y_pred - y) / n

def cross_entropy_loss(y, y_pred):
    epsilon = 1e-5
    return -1*np.sum(y * np.log(y_pred+epsilon)) / y.shape[0]

def cross_entropy_loss_prime(y, y_pred):
    return (y_pred-y) / y.shape[0]

# convulation layer
class Convulation(Layer):
    def __init__(self, output_channel, filter_size, input_shape=None, stride=1, padding=0):
        # output channel : number of kernels
        # input shape : image depth * height * width
        # filter size : 2d shape of kernel (d x d)

        
        self.output_channel = output_channel
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_channel = 0
        self.output_height = 0
        self.output_width = 0

        self.kernel = None
        self.bias = None

        if input_shape is not None:
            # print(input_shape)
            depth, height, width = input_shape
            self.input_channel = depth
            self.output_height = (height + 2*padding - filter_size[0]) // stride + 1
            self.output_width = (width + 2*padding - filter_size[1]) // stride + 1

            self.kernel = np.random.randn(output_channel,depth,filter_size[0], filter_size[1])
            self.bias = np.random.randn(output_channel,self.output_height, self.output_width)
    
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

    def convulate(self, input, kernel, stride=1, padding=0):
        if padding != 0:
            input = self.pad(input,padding)
        height = (input.shape[0]-kernel.shape[0])//self.stride+1
        width = (input.shape[1]-kernel.shape[1])//self.stride+1
        output = np.zeros((height,width))
        for h in range(0, height):
            i = h*self.stride
            for w in range(0, width):
                j = w*self.stride
                output[h][w] = np.sum(input[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
                # for k in range(kernel.shape[0]):
                #     for l in range(self.kernel.shape[1]):
                #         output[h][w] += input[i+k][j+l] * kernel[k][l]
            
        
        return output

    def forward_prop(self, input):
        # if input shape is not given in the constructor before then initialize kernel and bias
        if self.kernel is None:
            input_shape = input.shape
            depth, height, width = input_shape
            self.input_channel = depth
            self.output_height = (height + 2*self.padding - self.filter_size[0]) // self.stride + 1
            self.output_width = (width + 2*self.padding - self.filter_size[1]) // self.stride + 1

            self.kernel = np.random.randn(self.output_channel,depth,self.filter_size[0], self.filter_size[1]) *np.sqrt(2 / (self.filter_size[0] * self.filter_size[1] * depth))
            self.bias = np.random.randn(self.output_channel,self.output_height, self.output_width)
            # print('covvvv')
            # print(self.kernel)
            # print(self.bias)
        
        # main forward propagation
        self.input = input
        tmp_input = self.pad(input, self.padding)
        self.output = self.bias
        for i in range(self.output_channel):
            for j in range(tmp_input.shape[0]):
                self.output[i] += self.convulate(tmp_input[j],self.kernel[i][j])
        # self.output = np.einsum('j,ij->ihw')
        
        # for i in range(self.output_channel):
        #     self.output[i] += self.convulate(tmp_input,self.kernel[i])
        
        # print(self.bias.shape)
        # print(self.output)
        return self.output
    
    def backward_prop(self, gradient, learning_rate):

        wrt_kernel = np.zeros(self.kernel.shape)
        wrt_input = np.zeros(self.input.shape)

        kernel_gradient = self.changeGradient(gradient,padding=False)
        input_gradient = self.changeGradient(gradient)
        # print('convvv')
        # print(self.input)
        # print(kernel_gradient)
        # print(input_gradient)

        for i in range(self.output_channel):
            for j in range(self.input.shape[0]):

                wrt_kernel[i,j] = self.convulate(self.input[j],kernel_gradient[i])
                wrt_input[j] += self.convulate(input_gradient[i], np.rot90(self.kernel[i][j], 2))

        # rot_kernel = np.zeros(self.kernel.shape)
        # for i in range(self.output_channel):
        #     rot_kernel[i] = np.rot90(self.kernel[i], 2)
        # for i in range(self.output_channel):
        #     wrt_kernel[i] = self.convulate(self.input,kernel_gradient)
        #     wrt_input += self.convulate(input_gradient[i], rot_kernel[i])
                
        
        self.kernel -= learning_rate * wrt_kernel
        self.bias -= learning_rate * gradient

        return wrt_input

    def show(self):
        print('conv')
        print(self.kernel)
        print(self.bias)
        

# pooling layer

class Maxpool(Layer):
    def __init__(self, filter_size, stride=1, input_shape=None):
        self.filter_size = filter_size
        self.output_shape = None
        self.output = None
        self.stride = stride
        if input_shape is not None:
            depth, height, width = input_shape
            out_height = (height - filter_size[0]) // stride + 1
            out_width = (width - filter_size[1]) // stride + 1

            self.output_shape = (depth, out_height, out_width)
            self.output = np.zeros(self.output_shape)

    def forward_prop(self, input):
        if self.output is None:
            input_shape = input.shape
            depth, height, width = input_shape
            out_height = (height - self.filter_size[0]) // self.stride + 1
            out_width = (width - self.filter_size[1]) // self.stride + 1

            self.output_shape = (depth, out_height, out_width)
            self.output = np.zeros(self.output_shape)
        
        self.input = input
        depth, height, width = self.output_shape
        for ch in range(depth):
            for i in range(height):
                ii = i*self.stride
                for j in range(width):
                    jj = j*self.stride
                    self.output[ch,i,j] = np.max(self.input[ch, ii:ii+self.filter_size[0], jj:jj+self.filter_size[1]])
        return self.output
    
    def backward_prop(self, gradient, learning_rate):
        wrt_input = np.zeros(self.input.shape)
        depth, height, width = self.output_shape
        for ch in range(depth):
            for i in range(height):
                ii = i*self.stride
                for j in range(width):
                    jj = j*self.stride
                    val = np.max(self.input[ch, ii:ii+self.filter_size[0], jj:jj+self.filter_size[1]])
                    maxi, maxj = np.where(val == self.input[ch, ii:ii+self.filter_size[0], jj:jj+self.filter_size[1]])
                    wrt_input[ch, ii:ii+self.filter_size[0], jj:jj+self.filter_size[1]][maxi[0]][maxj[0]] = gradient[ch,i,j]
        return wrt_input
# flattening layer

class Flatten(Layer):
    def __init__(self, input_shape=None, output_shape=None):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_prop(self, input):
        self.input_shape = input.shape
        self.output_shape = (input.size,1)

        self.input = input
        self.output = np.reshape(input, self.output_shape)
        return self.output
    
    def backward_prop(self, gradient, learning_rate):
        return np.reshape(gradient, self.input_shape)


class Neural_Network():
    def __init__(self, epochs, loss, loss_prime, learning_rate, batch):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_prime = loss_prime
        self.layers = []
        self.batch_size = batch

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X_train, Y_train, X_val, Y_val):
        train_loss = []
        validation_loss = []
        f_score = []
        n_mini_batch = len(X_train) // self.batch_size
        for e in range(self.epochs):
            
            for _ in tqdm(range(n_mini_batch)):
                indices=list(range(len(X_train)))
                np.random.shuffle(indices)
                X = X_train[indices[0:self.batch_size]]
                Y = Y_train[indices[0:self.batch_size]]
                # print('epoch :',e)
                error = 0
                pred = []
                for x,y in zip(X,Y):
                    y_pred = x
                    y = y.T
                    # print(x.shape)
                    # print(y.shape)
                    # print('---------------------------------------------------------')
                    # print(y_pred.shape)
                    # print(y_pred)
                    for layer in self.layers:
                        y_pred = layer.forward_prop(y_pred)
                        # print('y-predddddddddd')
                        # print(y_pred.shape)
                        # print(y_pred)
                    error += self.loss(y, y_pred)
                    pred.append(y_pred.T)
                    # true.append(np.argmax())

                    # print('asdfsfsdfsdfsdfsdfsfsdfsfsfsfsfsd')
                    # print(y_pred)
                    # print(y)

                    # gradient = self.loss_prime(y,y_pred)
                    # print('start',gradient.shape)
                    
                    gradient = y_pred - y

                    # print(',-------------------------------------------------------------')
                    # print(gradient.shape)
                    # print(gradient)
                    
                    for layer in reversed(self.layers):
                        gradient = layer.backward_prop(gradient, self.learning_rate)
                        # print('gradieeeeeeeeent')
                        # print(gradient.shape)
                        # print(gradient)

                pred = np.array(pred)
                pred = np.argmax(pred, axis=2)
                Y = np.argmax(Y, axis=2)

                train_loss.append(accuracy_score(Y, pred))
                f_score.append(f1_score(Y,pred,average='macro'))

                error /= len(X)
                # for layer in self.layers:
                #     print(layer.show())
                #     layer.print_weight()
            if e % 1 == 0:
                print('epoch',e+1,':',error)
        return train_loss,f_score
    
    def predict(self, X):
        y_pred = X
        for layer in self.layers:
            y_pred = layer.forward_prop(y_pred)
        return y_pred
        
    def show(self):
        for l in self.layers:
            l.show()

def run(X,Y):
    indices=list(range(len(X)))
    np.random.shuffle(indices)

    ind=int(len(indices)*0.70)
    # train data
    X_train=X[indices[:ind]] 
    Y_train=Y[indices[:ind]]
    # validation data
    X_val=X[indices[-(len(indices)-ind):]] 
    Y_val=Y[indices[-(len(indices)-ind):]]
    # X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    # Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
    epochs = 1
    lr = 0.0001
    model = Neural_Network(epochs, loss = cross_entropy_loss, loss_prime = cross_entropy_loss_prime, learning_rate=lr, batch=64)
    model.add(Convulation(6, (5,5)))
    model.add(Relu())
    model.add(Maxpool((2,2)))
    model.add(Convulation(16, (5,5)))
    model.add(Relu())
    model.add(Maxpool((2,2)))
    model.add(Flatten())
    model.add(DenseLayer(120))
    model.add(DenseLayer(84))
    model.add(DenseLayer(10))
    model.add(Softmax())
    # model.add(Convulation(32, (5,5)))
    # model.add(Relu())
    # model.add(Convulation(32, (5,5)))
    # model.add(Relu())
    # model.add(Maxpool((2,2)))
    # model.add(Convulation(128, (3,3)))
    # model.add(Relu())
    # model.add(Convulation(128, (3,3)))
    # model.add(Relu())
    # model.add(Maxpool((2,2)))
    # model.add(Convulation(256, (3,3)))
    # model.add(Relu())
    # model.add(Maxpool((2,2)))

    # model.add(Flatten())
    # model.add(DenseLayer(64))
    # model.add(Relu())
    # model.add(DenseLayer(10))
    # model.add(Softmax())
    
    # model.add(Convulation(1, (3,3)))
    # model.add(Relu())
    # model.add(Maxpool((2,2)))

    # model.add(Flatten())
    # model.add(DenseLayer(5))
    # model.add(Relu())
    # model.add(DenseLayer(3))
    # model.add(Softmax())



    t_loss, f_score = model.fit(X_train, Y_train, X_val, Y_val)
    np.savetxt('train_loss.txt', t_loss, delimiter=',')
    np.savetxt('f1_score.txt', f_score, delimiter=',')
    return model

    # x_test = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
    # y_pred = []
    # for x in x_test:
    #     y_pred.append(model.predict(x))
    # print(y_pred)

    # # # model.show()

    # points = []
    # for x in np.linspace(0, 1, 20):
    #     for y in np.linspace(0, 1, 20):
    #         z = model.predict(np.array([[x], [y]]))
    #         points.append([x, y, z[0,0]])

    # points = np.array(points)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
    # plt.show()


x = np.zeros((1,4,4))
for i in range(4):
    for j in range(4):
        x[0,i,j] = i+j+1
# print(x)
# poop = Convulation(2, (2,2))
# print(poop.forward_prop(x))
# t = poop.forward_prop(x)
# print(t)
# t = poop.backward_prop(t,0.1)
# print(t)
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
        ret,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
        if dim_size is not None:
            img=cv2.resize(img,(dim_size,dim_size),interpolation=cv2.INTER_AREA)
            img = np.reshape(img, (1,dim_size,dim_size))
        x.append(img)
    x = np.array(x)
    y_label = data[:,1]
    y = np.zeros((y_label.shape[0],1,10))
    for i in range(y_label.shape[0]):
        y[i,0,y_label[i]] = 1 
    x = x.astype("float32")/255
    return (x,y)
filename_a = r'training-a.csv'
filename_b = r'training-b.csv'
filename_c = r'training-c.csv'
X_a,Y_a = load_data(filename_a,dim_size=28)
X_b,Y_b = load_data(filename_b,dim_size=28)
X_c,Y_c = load_data(filename_c,dim_size=28)
X = np.concatenate((X_a,X_b,X_c), axis = 0)
Y = np.concatenate((Y_a,Y_b,Y_c), axis = 0)

X=X[0:2000,:,:,:]
Y=Y[0:2000,:,:]
# X = np.arange(25)
# X = np.reshape(X,(1,1,5,5))
# new_x = np.zeros((3,1,5,5))
# for i in range(3):
#     new_x[i] = X +i+ 10
# X = new_x

# print(X[0])
# conv = Convulation(1,(3,3))
# print(conv.forward_prop(X[0]))
# print(X)
# Y = np.array([[[0,1,0],[1,0,0],[0,0,1]]])
# Y = np.reshape(Y,(3,1,3))
# print(X.shape)
# print(Y.shape)
# # print(X)
# # print(Y)
model = run(X,Y)
pickle.dump(model,open('model.pkl', 'wb'))

# model = pickle.load(open('model.pkl', 'rb'))
# X = X[100:120,:,:,:]
# Y = Y[100:120,:]
# y_pred = []

# for x in X:
#     y_pred.append(model.predict(x))
# # print(y_pred)
# y_pred = np.array(y_pred)
# y_pred = np.argmax(y_pred, axis=1)

# Y = np.argmax(Y, axis=2)
# print(y_pred)
# print(Y)

