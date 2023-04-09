import numpy as np
import os
import gzip
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt

'''
激活函数
'''
def Relu(x):
    return np.maximum(0, x)


def Derivative_Of_Relu(x):
    return x > 0


def Softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    normlize = np.sum(x)
    return x/normlize

'''
数据预处理以及读取数据
'''
def vector(num):
    vec = np.zeros((10, 1))
    vec[num] = 1.0
    return vec

def data_process(data):
    res = []
    data_x = [np.reshape(x, (784, 1)) for x in data[0]] #train_data[0]中的每个元素都是28*28的0-1矩阵
    data_y = [vector(y) for y in data[1]]  #举例：train_data[1]=[0,1,5,2,2]
    res = list(zip(data_x, data_y))   #zip打包好会返回一个对象 所以使用list转换成列表
    return res

def load_data():
    file = gzip.open(os.path.join(os.curdir, "datasets", "mnist.pkl.gz"), "rb")  #curdir\datasets\mnist.pkl.gz
    train_data, val_data, test_data = pickle.load(file, encoding="latin1") #Latin1编码是单字节编码，向下兼容ASCII
    file.close()
    
    train_data = data_process(train_data)
    val_data = data_process(val_data)
    test_data = data_process(test_data)
    
    return train_data, val_data, test_data

'''
模型保存以及加载
'''
def save(model, filename):
    np.savez_compressed(
        file=os.path.join(os.curdir, 'models', filename),
        weights=model.weights,
        biases=model.biases,
        linear_transforms=model.linear_transforms,
        activations=model.activations
    )
    
def load(model, filename):
    npz_load = np.load(os.path.join(os.curdir, 'models', filename), allow_pickle=True)
    model.weights = list(npz_load['weights'])
    model.biases = list(npz_load['biases'])
    model.sizes = [b.shape[0] for b in model.biases]
    model.num_layers = len(model.sizes)
    model.linear_transforms = list(npz_load['linear_transforms'])
    model.activations = list(npz_load['activations'])
    return model

'''
模型训练及测试相关
'''
def test(model, test_data):      #测试集的准确率
    res = []
    for x, label in test_data:
        output = model.forward(x)
        res.append(np.argmax(output) == np.argmax(label))
    accuracy = sum(res) / 100.0
    print(f"********测试集准确率是:{accuracy} %.********")

def fit(model, optimizer, training_data, val_data, epochs):
    best_accuracy = 0
    train_losses = []
    val_losses = []
    accuracies = []
    for epoch in range(epochs):
        # validate
        val_loss = 0
        res = []
        for x, label in val_data:
            output = model.forward(x)
            val_loss += np.where(label==1, -np.log(output), 0).sum() #当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
            res.append(np.argmax(output) == np.argmax(label))  #output最大的那个索引等于label的索引
        val_loss /= len(val_data)
        val_losses.append(val_loss)
        accuracy = sum(res) / 100.0
        accuracies.append(accuracy)
        # train   
        random.shuffle(training_data)  
        batches = [training_data[k:k+optimizer.batch_size] for k in range(0, len(training_data), optimizer.batch_size)]
        train_loss = 0
        for batch in batches:
            optimizer.zero_grad()
            for x, label in batch:
                output = model.forward(x)
                # loss为交叉熵损失函数
                train_loss  += np.where(label==1, -np.log(output), 0).sum()
                Derivative_Of_loss = output - label
                delta_nabla_b, delta_nabla_w = model.backward(Derivative_Of_loss)
                optimizer.update(delta_nabla_b, delta_nabla_w)
            optimizer.step()
        train_loss /= len(training_data)
        train_losses.append(train_loss)
        # save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save(model,f"model_{model.sizes[1]}_{optimizer.lr}_{optimizer.L2_Regularization}.npz")
        print(f"****Epoch {epoch+1}, accuracy {accuracy} %.")
    # save  每个epoch都有保存
    data = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": accuracies
    }
    pd.DataFrame(data).to_csv(f'records/record_{model.sizes[1]}_{optimizer.lr}_{optimizer.L2_Regularization}.csv',)
    return best_accuracy


'''
参数可视化
'''
def figure(model, record):
    # 可视化训练和测试的loss曲线
    plt.subplot(2,3,1)
    plt.plot(record[['train_loss','val_loss']])
    plt.title('train/val loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")

    # 可视化测试的accuracy曲线
    plt.subplot(2,3,2)
    plt.plot(record[['val_accuracy']])
    plt.title('val accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    # 可视化每层的网络参数
    plt.subplot(2,3,3)
    hiddenlayer_weights = model.weights[1].flatten().tolist() #flatten()和flatten(0)一样，全部展开
    plt.hist(hiddenlayer_weights, bins=100)
    plt.title("hiddenlayer weights")
    plt.xlabel("value")
    plt.ylabel("frequency")
    
    plt.subplot(2,3,4)
    hiddenlayer_biases = model.biases[1].flatten().tolist()
    plt.hist(hiddenlayer_biases, bins=10)
    plt.title("hiddenlayer biases")
    plt.xlabel("value")
    plt.ylabel("frequency")

    plt.subplot(2,3,5)
    outputlayer_weights = model.weights[2].flatten().tolist()
    plt.hist(outputlayer_weights, bins=20)
    plt.title("outputlayer weights")
    plt.xlabel("value")
    plt.ylabel("frequency")
    
    plt.subplot(2,3,6)
    outputlayer_biases = model.biases[2].flatten().tolist()
    plt.hist(outputlayer_biases, bins=10)
    plt.title("outputlayer biases")
    plt.xlabel("value")
    plt.ylabel("frequency")
    
    plt.tight_layout()
    plt.savefig("figs/figure.png")
    plt.close()
