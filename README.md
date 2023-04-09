22210980055 刘晋汐

python main.py 运行

1. 代码文件
- main.py
  - 训练
  - 参数查找
  - 测试
- model.py
  - 模型初始化
  - 前向传播
  - 反向传播
- optimizer.py
  - 模型参数梯度初始化
  - 更新参数
- utils.py
  - 工具函数
- datasets
  - 保存数据集
- records
  - 保存用于可视化的数据
- figs
  - 保存可视化图像
- models
  - 保存模型
2. 数据
- 数据集：MINIST
  - 下载初始数据集，其中训练集数据个数60000，测试集个数10000
- 数据预处理
  - 加载图片数据，转化为numpy格式，每张图片大小为28*28，将图片转换成(784,)的向量，图片标签转换成(10,)的向量，并按5：1将训练集分为train_data和val_data，并将train_data,val_data,test_data保存到pkl文件中（该部分直接使用已有pkl文件，仅描述过程）
  - 加载datasets文件夹中的pkl文件获取train_data,val_data,test_data

3. 训练
- 输入层 - 隐藏层 - 输出层，层与层之间全连接
  - 输入层784个神经元，输出层10个神经元
- 前向传播forward
  - 输入层经过线性变换获得线性变换的值，通过激活函数ReLU/Softmax分别获取隐藏层和输出层的值，最终forward函数返回输出层的值
  - 激活函数
    -采用ReLU用于隐层神经元输出
    -采用Softmax用于多分类神经网络输出
- 反向传播backward
  - 在反向传递时，计算每一层的输入的梯度（最后一层利用目标结果计算梯度）
- 参数设置
  - 学习率分别设置为1e-2，5e-3，1e-3(学习率下降)
  - 隐藏层神经元个数分别设置为20，30，40
  - 正则化强度分别设置为0，1e-4，5e-4（L2正则化）
- 模型参数梯度初始化为0
  - zero_grad()
- 模型参数更新
  - update()
  - step()
4. 模型保存及测试
- 参数查找
- 导入最佳模型模型
- 在测试集上的分类精度

5. 可视化
- 数据可视化
  - 训练的loss曲线
  - 测试的loss曲线
  - 测试的accuracy曲线
  - 隐藏层权重
  - 隐藏层偏置
  - 输出层权重
  - 输出层偏置
6. 链接
- 代码链接：https://github.com/Jarvis-JX/assignment
- 模型链接：https://pan.baidu.com/s/1Pj3PSarv-4FkL_oc_c-lyg
  - 提取码：4j6k
