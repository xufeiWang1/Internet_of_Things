#客户端
#调用load
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from load import *  #   load.py 里定义了函数getdata，用于边缘端数据划分，all_labels_train和all_features_train就是要用的

# import numpy as np
# import pandas as pd
# from sklearn import  preprocessing
# import warnings

class ClientsGroup(object):  #  构造边缘端集合类
    def __init__(self, dev, class_num):
        self.dev = dev
        self.clients_set = {}
        self.class_num = class_num
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):  #  初始化集合的内容
        index_class = getdata(self.class_num)  #  getdata是load.py里的函数
        # 获取数据的行数
        num_rows = index_class.shape[0]     # =3000

        # 打乱行索引的顺序
        shuffled_indices = np.random.permutation(num_rows)

        # 根据打乱后的行索引重新排列数组
        shuffled_array = index_class[shuffled_indices]

        # 重新变成 10 个 300×14 维的数组
        reshaped_array = shuffled_array.reshape(10, 300, 14)

        for i in range(self.class_num):
            self.clients_set[i].train_ds = reshaped_array[i]



            #someone_1 = client(TensorDataset(torch.tensor(local_data, dtype=torch.float, requires_grad = True), torch.tensor.....))


            #self.clients_set['client{}'.format(i)] = someone_1


class client(object):  #  构造每个边缘类
    def __init__(self, trainDataSet, dev, num_example):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.num_example = num_example
        self.state = {}


    def localUpdate(self, localBatchSize, localepoch, Net, lossFun, opti, global_parameters):  #  本地计算函数
        global_parameters = torch.load("global_parameters.path")
        Net.load_state_dict(global_parameters, strict = True)
        self.train_dl = DataLoader(self.train_ds, batch_size = localBatchSize, shuffle = True)
        #for epoch in range(localepoch):
        torch.save(Net.state_dict(), "global_parameters.pth")



dev = torch.device("cpu")
myClients = ClientsGroup(dev, 10)