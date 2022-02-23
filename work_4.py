import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi

plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号

# 数据准备
theta = np.arange(0, 16)
Bx = np.array([0, 9.1, 19.1, 27.3, 36.3, 42.5, 45.8, 44.5, 40.9, 36.1, 31, 26.5, 22.2, 18.5, 15.5, 13])

# 取样点的一半进行插值，这里取奇数点
x_inter = theta[::2]
y_inter = Bx[::2]

# 进行三次样条插值
ipo3 = spi.splrep(x_inter, y_inter, k=3)  # 样本点导入，生成参数
pp = spi.PPoly.from_spline(ipo3)

print("三次样条插值函数S(x)的插值节点为：\n %s" % pp.x)
print("三次样条插值函数S(x)在各段的三次函数各项系数：\n %s" % pp.c.T)

# 根据观测点和样条参数，生成插值
new_x = np.arange(0, 15, 0.1)
iy3 = spi.splev(new_x, ipo3)
yy = spi.splev(theta, ipo3)
print("各点误差为：\n %s" % (Bx - yy))

plt.xlim((0, 16))
plt.ylim((0, 50))
plt.plot(new_x, iy3, linewidth=1.5, label = '插值曲线')
plt.scatter(theta, Bx, 50, marker= '*', label = '样本点')
plt.legend(loc = 0)
plt.grid(True)
plt.title("传感器位置磁场强度x轴分量与磁钢转过角度的关系曲线")
plt.xlabel("θ(°)")
plt.ylabel("Bx(mT)")
plt.show()
