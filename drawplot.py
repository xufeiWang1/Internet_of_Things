import matplotlib.pyplot as plt
from server import *
from load import *
import pandas as pd
import torch.cuda
from matplotlib.dates import DateFormatter, YearLocator



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
plt.scatter(dates1, temperatures1, label = 'true temperature')
plt.scatter(dates1, temperatures2, label = 'predict temperature')


#   接下来画测试集预测曲线
testlable = pd.read_csv('test.csv')

# trainxlable.rename(columns={"\ufeffyear": "year"}, inplace=True)
# 提取年、月、日和温度数据
temperaturesTestTrue = np.array(testlable['actual'])


monthsTest = np.array(testlable['month'])
daysTest = np.array(testlable['day'])
yearsTest = np.array(testlable['year'])

# 创建日期字符串，包含年份、月份和日期
datesTest = [f"{int(year)}/{int(month)}/{int(day)}" for year, month, day in zip(yearsTest, monthsTest, daysTest)]
predict_test = getTestData()
x = torch.tensor(predict_test, dtype=torch.float)  # 验证拟合和预测结果
# x = x.to(dev)  # 将输入数据和目标数据移动到设备上
temperatures_test  = myClients.Net(x[:, :13])  # __call__()方法像函数一样调用对象
temperatures_test = temperatures_test.cpu()
# 将张量转换为NumPy数组
temperatures_test = temperatures_test.detach().numpy()
# 绘制折线图
plt.xticks([])
plt.scatter(datesTest, temperaturesTestTrue, label = 'true test temperature')
plt.scatter(datesTest, temperatures_test, label = 'predict test temperature')

# 设置横轴刻度为年份


# 添加标题和轴标签
plt.title('fitting results')
plt.xlabel('Year')
plt.ylabel('Temperature')
# 添加图例
plt.legend()
# 显示图形
plt.show()


