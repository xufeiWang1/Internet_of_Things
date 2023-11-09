import matplotlib.pyplot as plt
from server import *
from load import *
import pandas as pd
import torch.cuda
from matplotlib.dates import DateFormatter, YearLocator



# 假设你的数据存储在一个名为data的二维数组中，形状为(3000, 4)，其中每列分别表示年、月、日和温度
trainxlable = pd.read_csv('train.csv')

trainxlable.rename(columns={"\ufeffyear": "year"}, inplace=True)
# 提取年、月、日和温度数据
temperatures1 = np.array(trainxlable['actual'])


months = np.array(trainxlable['month'])
days = np.array(trainxlable['day'])
years = np.array(trainxlable['year'])

# 创建日期字符串，包含年份、月份和日期
dates1 = [f"{int(year)}/{int(month)}/{int(day)}" for year, month, day in zip(years, months, days)]




predict_data = getdata()
x = torch.tensor(predict_data, dtype=torch.float)  # 验证拟合和预测结果
# x = x.to(dev)  # 将输入数据和目标数据移动到设备上
temperatures2  = myClients.Net(x[:, :13])  # __call__()方法像函数一样调用对象
temperatures2 = temperatures2.cpu()

# 将张量转换为NumPy数组
temperatures2 = temperatures2.detach().numpy()
# 绘制折线图
plt.xticks([])
plt.plot(dates1, temperatures1, label = 'true temperature')
plt.plot(dates1, temperatures2, label = 'predict temperature')

# 设置横轴刻度为年份


# 添加标题和轴标签
plt.title('Temperature Variation')
plt.xlabel('Year')
plt.ylabel('Temperature')
# 添加图例
plt.legend()
# 显示图形
plt.show()

# # 创建示例数据
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
#
# # 绘制折线图
# plt.plot(x, y)
#
# # 添加标题和轴标签
# plt.title('过去的真实温度和拟合温度')
# plt.xlabel('日期')
# plt.ylabel('最高温度（F：华氏）')
#
#
# # 显示图形
# plt.show()
#
#
#
#
# plt.plot(x, y)
#
# # 添加标题和轴标签
# plt.title('未来的真实温度和拟合温度')
# plt.xlabel('日期')
# plt.ylabel('最高温度（F：华氏）')
#
# # 显示图形
# plt.show()