import numpy as np
import math
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 共轭梯度法
def cg(A, b, x, r, d, epsilon):
    while np.linalg.norm(r) >= epsilon:
        temp = np.linalg.norm(r) ** 2
        alpha = np.dot(d.T, A)
        alpha = np.dot(alpha, d)
        alpha = temp / alpha
        x += alpha * d
        r = b - np.dot(A, x)
        beta = np.linalg.norm(r) ** 2 / temp
        d = r + beta * d


# xi,yi是给定数据，p是拟合多项式的次数
def my_polyfit(xi, yi, p):
    # 构造法方程组
    m = len(xi)
    n = p + 1
    err = 0

    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(m):
                G[i][j] += pow(xi[k], i + j)

    y = np.zeros(n)
    for i in range(n):
        for k in range(m):
            y[i] += pow(xi[k], i) * yi[k]

    # 采用共轭梯度法求解法方程组
    c0 = np.zeros(n)
    c0 = c0.reshape(-1, 1)
    c = c0

    y = y.reshape(-1, 1)
    r0 = y - np.dot(G, c0)
    r = r0
    d = r0

    cg(G, y, c, r, d, 1e-8)

    # 输出拟合多项式的各项系数
    print('拟合多项式的各项系数为：')
    for i in range(len(c)):
        print('c', i, '= ', c[i], sep='')

    # 计算拟合多项式的误差
    for i in range(m):
        temp = 0
        for j in range(n):
            temp += c[j] * pow(xi[i], j)
        err += pow(temp - yi[i], 2)
    err = math.sqrt(err)
    print('拟合多项式的误差E=', err)

    # 作出拟合多项式的曲线
    xt = np.linspace(xi[0], xi[-1], len(xi) * 20)
    yt = 0
    for i in range(len(c)):
        yt += pow(xt, i) * c[i]
    plt.title("最小二乘拟合四次多项式曲线图")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xi, yi, '*')
    plt.plot(xt, yt)
    plt.show()


if __name__ == "__main__":
    # 拟合数据
    xi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    yi = [5.1234, 5.3057, 5.5687, 5.9375, 6.4370, 7.0978, 7.9493, 9.0253, 10.3627]

    # 调用最小二乘拟合多项式函数，参数为数据点xi，yi和拟合的多项式次数
    my_polyfit(xi, yi, 3)
