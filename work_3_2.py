import numpy as np
import math
import sympy as sy


# 构造第(1)题的非线性方程组中的f(x)
def fx1():
    f1 = x1 ** 2 + x2 ** 2 + x3 ** 2 - 1
    f2 = 2 * x1 ** 2 + x2 ** 2 - 4 * x3
    f3 = 3 * x1 ** 2 - 4 * x2 ** 2 + x3 ** 2
    return sy.Matrix([f1, f2, f3])


# 计算第(1)题的f(x)
def cal_fx1(x):
    temp = fx1().subs([(x1, x[0]), (x2, x[1]), (x3, x[2])])
    temp = np.mat(temp).reshape(-1, 1)
    return temp


# 构造第(1)题f(x)的雅克比矩阵
def J_fx1():
    return sy.Matrix([sy.diff(fx1(), x1),
                      sy.diff(fx1(), x2),
                      sy.diff(fx1(), x3)])


# 计算第(1)题f(x)的雅克比矩阵
def cal_J_fx1(x):
    temp = np.mat(J_fx1().subs([(x1, x[0]), (x2, x[1]), (x3, x[2])]))
    temp = np.reshape(temp, (3, 3)).T
    temp = np.array(temp, dtype=float)
    return temp


# 构造第(2)题的非线性方程组中的f(x)
def fx2():
    f1 = sy.cos(x1 ** 2 + 0.4 * x2) + x1 ** 2 + x2 ** 2 - 1.6
    f2 = 1.5 * x1 ** 2 - 1 / 0.36 * x2 ** 2 - 1
    return sy.Matrix([f1, f2])


# 计算第(2)题的f(x)
def cal_fx2(x):
    temp = fx2().subs([(x1, x[0]), (x2, x[1])])
    temp = np.mat(temp).reshape(-1, 1)
    return temp


# 构造第(2)题f(x)的雅克比矩阵
def J_fx2():
    return sy.Matrix([sy.diff(fx2(), x1),
                      sy.diff(fx2(), x2)])


# 计算第(2)题f(x)的雅克比矩阵
def cal_J_fx2(x):
    temp = np.mat(J_fx2().subs([(x1, x[0]), (x2, x[1])]))
    temp = np.reshape(temp, (2, 2)).T
    temp = np.array(temp, dtype=float)
    return temp


# 牛顿法
def Newton_set(f, J_f, x, epsilon, k):
    n = eval(f)(x).size

    for i in range(k):
        fx = eval(f)(np.array(x).flatten())
        J_fx = eval(J_f)(np.array(x).flatten())
        dx = np.dot(np.linalg.inv(J_fx), -fx)
        error = math.sqrt(np.dot(dx.T, dx) / np.dot(x.T, x))
        x = x + dx
        if error < epsilon:
            return [x, error, i + 1]

    return [x, error, i + 1]


# 弦割法
def Secant_set(f, J_f, x, epsilon, h, k):
    n = eval(f)(x).size
    f_ij = np.zeros((n, n))

    for i in range(k):
        temp = np.array(x)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1
            e = e.reshape(-1, 1)
            f_ij[:, j] = eval(f)(np.array(x + h * e).flatten()).flatten()
        f_i = eval(f)(np.array(x).flatten())
        z = np.dot(np.linalg.inv(f_ij), f_i)
        x = x + h * z / (z.sum() - 1)
        error = np.linalg.norm(np.array(x - temp, dtype=float))
        if error < epsilon:
            return [x, error, i + 1]

    return [x, error, i + 1]


# 布洛依登法
def Broyden_set(f, J_f, x, epsilon, k):
    n = eval(f)(x).size
    A = eval(J_f)(np.array(x).flatten())
    A_inv = np.linalg.inv(A)
    x0 = x
    fx0 = eval(f)(np.array(x0).flatten())
    x1 = x0 - A_inv * fx0

    for i in range(k):
        s = x1 - x0
        fx0 = eval(f)(np.array(x0).flatten())
        fx1 = eval(f)(np.array(x1).flatten())
        y = fx1 - fx0
        A_inv = A_inv + (s - A_inv * y) * s.T * A_inv / (1 + s.T * A_inv * y)
        x2 = x1 - A_inv * fx1
        error = np.linalg.norm(np.array(x2 - x1, dtype=float))
        x1, x0 = x2, x1
        if error < epsilon:
            return [x2, error, i + 1]

    return [x, error, i + 1]


if __name__ == "__main__":
    epsilon: float = 1e-8  # 误差
    k = 100  # 最大迭代次数
    h = 0.01  # 弦割法中的h
    x1, x2, x3 = sy.symbols('x1 x2 x3')  # 定义符号x1, x2, x3

    print("下面求解7.3(1):\n")

    x0 = np.array([1.0, 1.0, 1.0], dtype=float).reshape(-1, 1)  # 给定第(1)题的初始向量

    [x, error, times] = Newton_set("cal_fx1", "cal_J_fx1", x0, epsilon, k)
    print("采用牛顿法求解非线性方程组：")
    print("x = %s" % x)
    print("误差：%s" % error)
    print("迭代次数：%s" % times)

    [x, error, times] = Secant_set("cal_fx1", "cal_J_fx1", x0, epsilon, h, k)
    print("\n采用弦割法求解非线性方程组：")
    print("x = %s" % x)
    print("误差：%s" % error)
    print("迭代次数：%s" % times)

    [x, error, times] = Broyden_set("cal_fx1", "cal_J_fx1", x0, epsilon, k)
    print("\n采用布洛依登法求解非线性方程组：")
    print("x = %s" % x)
    print("误差：%s" % error)
    print("迭代次数：%s" % times)

    print("\n下面求解7.3(2):")

    x0 = np.array([1.04, 0.47]).reshape(-1, 1)  # 给定第(2)题的初始向量

    [x, error, times] = Newton_set("cal_fx2", "cal_J_fx2", x0, epsilon, k)
    print("\n采用牛顿法求解非线性方程组：")
    print("x = %s\n" % x)
    print("误差：%s" % error)
    print("迭代次数：%s" % times)

    [x, error, times] = Secant_set("cal_fx2", "cal_J_fx2", x0, epsilon, h, k)
    print("\n采用弦割法求解非线性方程组：")
    print("x = %s\n" % x)
    print("误差：%s" % error)
    print("迭代次数：%s" % times)

    [x, error, times] = Broyden_set("cal_fx2", "cal_J_fx2", x0, epsilon, k)
    print("\n采用布洛依登法求解非线性方程组：")
    print("x = %s\n" % x)
    print("误差：%s" % error)
    print("迭代次数：%s" % times)
