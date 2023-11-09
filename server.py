import os
import argparse
import torch
import torch.cuda
from tqdm import tqdm  # 用于显示进度条
from torch import optim  # 包括常用的优化算法如SGD,Adam，Adagrad等
from client import  ClientsGroup  #
import sys
from Model import tianqi_2NN
from load import *


#   设置全局参数
args = {
    'num_of_clients': 10,
    'num_comn': 10,
    'batchsize': 32,
    'epoch': 5,
    'learning_rate': 0.001,
    'input_size': 13,
    'hidden_size': 5,
    'output_size': 1
}



net = tianqi_2NN(args['input_size'], args['hidden_size'], args['output_size'])  # 实例化神经网络
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net)

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

net = net.to(dev)

loss_func = torch.nn.MSELoss(reduction='mean')  # 确定损失函数和优化器
opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

myClients = ClientsGroup(dev, args['num_of_clients'], net)  # 实例化边缘端集合对象
myClients.client_create()
myClients.initnet()
for i in range(1, args['num_comn'] + 1):  # 边缘端计算
    print('-------------------------fedavg----------------------')
    print('------------------------------第', i, '次训练--------------------')
    myClients.dataSetBalanceAllocation()
    myClients.updateSet(net, loss_func)
    myClients.combineParameters()
    myClients.send_parameter()

test_data = getTestData()
x = torch.tensor(test_data, dtype=torch.float)  # 验证拟合和预测结果
x = x.to(dev)  # 将输入数据和目标数据移动到设备上
predict_1 = net(x[:, :13])  # __call__()方法像函数一样调用对象
predict = predict_1.cpu().detach().numpy()
loss_test = loss_func(predict_1, x[:, 13])
print(f" Loss: {loss_test.item()}")
