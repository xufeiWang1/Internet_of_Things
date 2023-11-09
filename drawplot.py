import matplotlib.pyplot as plt
from server import *

# 创建示例数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 绘制折线图
plt.plot(x, y)

# 添加标题和轴标签
plt.title('过去的真实温度和拟合温度')
plt.xlabel('日期')
plt.ylabel('最高温度（F：华氏）')


# 显示图形
plt.show()




plt.plot(x, y)

# 添加标题和轴标签
plt.title('未来的真实温度和拟合温度')
plt.xlabel('日期')
plt.ylabel('最高温度（F：华氏）')

# 显示图形
plt.show()