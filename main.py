import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义参数和中心点
radius = 6371393
center = [0, 0, 0]

# 生成参数的网格
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)

# 计算 x、y、z 坐标
x = radius * np.cos(u) * np.sin(v) + center[0]
y = radius * np.sin(u) * np.sin(v) + center[1]
z = radius * np.cos(v) + center[2]

# 创建图形并绘制曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='blue')

# 绘制要可视化的点
point = np.array([-3900408.56738947, -5657106.57208116, 0])
ax.scatter(*point, color='red')

# 画轨道
radius = 6871393.0
center = (0, 0)
theta = np.linspace(0, 2*np.pi, 100)
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)
z = 0
ax.plot(x, y, color='red')

# 显示图形
plt.show()



