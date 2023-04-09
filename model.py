import numpy as np
import os
import utils


class NeuralNetwork:
    
    def __init__(self, sizes):
        '''
        初始化神经网络
        '''
        self.sizes = sizes
        self.num_layers = len(sizes)
        
        ''' 
        初始化权重 当sizes是[784，20，10]时， zip(sizes[1:], sizes[:-1])=[(20,784),(10,20)]   这样第一个的话y就是20，x就是784
        最后weights的len其实是3，第一层是0，后面的都是靠生成的
        [[第一个],[...],[第y个]]
        存储线性变换和非线性变换的结果的结果  bias.shape分别是(1,)，(20, 1)，(10, 1)
        '''      
        self.weights = [np.array([0])] + [np.random.randn(y, x)/np.sqrt(x) for y, x in zip(sizes[1:], sizes[:-1])] #
        self.biases = [np.array([0])] + [np.random.randn(x, 1) for x in sizes[1:]]
        self.linear_transforms = [np.zeros(bias.shape) for bias in self.biases]
        self.activations = [np.zeros(bias.shape) for bias in self.biases]
    
    def forward(self, x):  
        '''
        前向传播   x是28*28
        '''
        self.activations[0] = x
        for i in range(1, self.num_layers): 
            '''
            线性变换        weights一共有三个 
            非线性变换
            在前面使用Relu，在最后一层使用softmax
            '''
            self.linear_transforms[i] = self.weights[i].dot(self.activations[i-1]) + self.biases[i]
            self.activations[i] = utils.Softmax(self.linear_transforms[i]) if i == self.num_layers-1 else utils.Relu(self.linear_transforms[i])

        return self.activations[-1]  #返回最后一层
    
    def backward(self, Derivative_Of_loss):
        '''
        反向传播   

        `Derivative_Of_loss为损失函数的求导结果
        '''
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        nabla_b[-1] = Derivative_Of_loss
        nabla_w[-1] = Derivative_Of_loss.dot(self.activations[-2].transpose())

        for layer in range(self.num_layers-2, 0, -1):
            Derivative_Of_loss = np.multiply(
                self.weights[layer+1].transpose().dot(Derivative_Of_loss),
                utils.Derivative_Of_Relu(self.linear_transforms[layer])
            )
            nabla_b[layer] = Derivative_Of_loss
            nabla_w[layer] = Derivative_Of_loss.dot(self.activations[layer-1].transpose())
        
        return nabla_b, nabla_w
    

