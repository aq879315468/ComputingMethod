import numpy as np
import math
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


# 梯度下降法
def gd(A, b, x, r, epsilon):
    r = b - np.dot(A, x)
    while np.linalg.norm(r) >= epsilon:
        temp = np.linalg.norm(r) ** 2
        alpha = np.dot(r.T, A)
        alpha = np.dot(alpha, r)
        alpha = temp / alpha
        x += alpha * r
        r = b - np.dot(A, x)
        error.append(math.log(np.linalg.norm(r), 10))


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
        error.append(math.log(np.linalg.norm(r), 10))


# 判断系数矩阵是否是对称正定矩阵
def Is_sym_pos(A):
    if not (np.transpose(A) == A).all():
        return 0
    eigen, feature = np.linalg.eig(A)
    for i in range(n):
        if eigen[i] < 0:
            return 0
    return 1


if __name__ == "__main__":
    epsilon: float = 1e-6
    n = int(input("请输入线性方程组的阶数: "))
    cal = int(input("计算P113的计算实习3.2请输入1，自选线性方程组Ax=b请输入0: "))
    error = []

    # 用户输入线性方程组
    if cal == 0:
        A = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(n):
            for j in range(n):
                A[i][j] = float(input("A 第%s行 第%s列：" % (i + 1, j + 1)))
        print("A = %s" % A)

        for i in range(n):
            b[i] = float(input("b 第%s个：" % (i + 1)))
            b = b.reshape(-1, 1)
        print("b = %s" % b)

        # 如果系数矩阵A不是对称正定矩阵，将Ax=b变为ATAx=ATb
        if Is_sym_pos(A) == 0:
            print("矩阵A不是对称正定矩阵")
            b = np.dot(np.transpose(A), b)
            print("修正b = %s" % b)
            A = np.dot(np.transpose(A), A)
            print("修正A = %s" % A)

    # 计算P113的计算实习3.2
    elif cal == 1:
        A = -2 * np.eye(n)
        A += np.eye(n, None, 1)
        A += np.eye(n, None, -1)

        b = np.zeros(n)
        b[0] = -1
        b[n - 1] = -1
        b = b.reshape(-1, 1)
    else:
        exit("输入错误")

    # 给定初始向量x0
    x0 = np.zeros(n)
    x0 = x0.reshape(-1, 1)
    x = x0

    r0 = b - np.dot(A, x0)
    r = r0
    d = r0

    method = int(input("采用共轭梯度法请输入1，采用梯度下降法请输入0: "))

    if method == 1:
        cg(A, b, x, r, d, epsilon)  # 采用共轭梯度法
    elif method == 0:
        gd(A, b, x, r, epsilon)  # 采用梯度下降法
    else:
        exit("输入错误")

    print("x^T = %s" % x.T)
    print("times = %s" % len(error))

    # 画出收敛速度图
    x2 = np.arange(1, len(error) + 1)
    plt.title("收敛速度图")
    plt.xlabel("迭代次数")
    plt.ylabel("误差取对数")
    plt.plot(x2, error[0:])
    plt.show()
