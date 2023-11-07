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
        for i, idcs in enumerate(index_class):  #  边缘数据划分
            #local_label, local_data = np.vstack(train_labels_shuffle[idcs]), np.vstack(train_features_shuffle[idcs])
            #num_example = len(local_label)

            #someone_1 = client(TensorDataset(torch.tensor(local_data, dtype=torch.float, requires_grad = True), torch.tensor.....))
            A = i
            B = idcs

            #self.clients_set['client{}'.format(i)] = someone_1


class client(object):  #  构造每个边缘类
    def __init__(self, trainDataSet, dev, num_example):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.num_example = num_example
        self.state = {}


    def localUpdate(self, localBatchSize, localepoch, Net, lossFun, opti, global_parameters):  #  本地计算函数
        Net.load_state_dict(global_parameters, strict = True)
        self.train_dl = DataLoader(self.train_ds, batch_size = localBatchSize, shuffle = True)
        #for epoch in range(localepoch):



dev = torch.device("cpu")
myClients = ClientsGroup(dev, 10)