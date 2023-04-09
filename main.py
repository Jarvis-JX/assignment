import numpy as np
import pandas as pd
import utils
from model import NeuralNetwork
from optimizer import SGD

np.random.seed(48)

batch_size = 32
epochs= 40
Networks = [[784, 20, 10], [784, 30, 10], [784, 40, 10]]
Learning_Rates = [1e-3, 5e-3, 1e-2]
L2_Regularizations = [0, 1e-4, 5e-4]

# batch_size = 32
# epochs= 3
# Networks = [[784, 30, 10]]
# Learning_Rates = [1e-3]  
# L2_Regularizations = [5e-4]

print('********加载MINIST数据集********')
train_data, val_data, test_data = utils.load_data()   #utils.py里

print('********模型训练********')
# 参数查找
final = {'accuracy':0.0}
for network in Networks:
    for learning_rate in Learning_Rates:
        for L2_Regularization in L2_Regularizations:
            print(f"********隐藏层大小: {network}, 学习率: {learning_rate}, 正则化强度: {L2_Regularization}********")
            model = NeuralNetwork(network)    #layer里面的[784,20,10]  就是sizes  784输入层  20隐含层  10输出层
            optimizer = SGD(model, learning_rate, L2_Regularization, batch_size)
            accuracy = utils.fit(model, optimizer, train_data, val_data, epochs)
            if accuracy > final['accuracy']:
                final['accuracy'] = accuracy
                final['network'] = network
                final['learning_rate'] = learning_rate
                final['L2_Regularization'] = L2_Regularization
print(f"********模型全部保存完成，最佳隐藏层大小为{final['network']}，最佳学习率为{final['learning_rate']}，最佳正则化强度为{final['L2_Regularization']}********")
      
# utils.save(model,f"model_{final['network'][1]}_{final['learning_rate']}_{final['L2_Regularization']}.npz")
                
print('********模型测试********')
model = NeuralNetwork(final['network'])
model = utils.load(model, f"model_{final['network'][1]}_{final['learning_rate']}_{final['L2_Regularization']}.npz")
utils.test(model, test_data)

print('********可视化********')
record = pd.read_csv(f"records/record_{final['network'][1]}_{final['learning_rate']}_{final['L2_Regularization']}.csv")
utils.figure(model, record)