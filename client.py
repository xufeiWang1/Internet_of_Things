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





# dev = torch.device("cpu")
# myClients = ClientsGroup(dev, 10)


#         Net.load_state_dict(global_parameters, strict = True)
#         global_parameters = torch.load("global_parameters.path")
#          torch.save(Net.state_dict(), "global_parameters.pth")