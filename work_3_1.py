import math
import sympy as sy
from sympy.abc import t


# 构造非线性方程中的f(x)
def f():
    fx = t ** 6 - 5 * t ** 5 + 3 * t ** 4 + t ** 3 - 7 * t ** 2 + 7 * t - 20
    return fx


# 构造f(x)的导数
def f_diff():
    fx_diff = sy.diff(f(), t)
    return fx_diff


# 计算f(x)
def cal_fx(x):
    return f().evalf(subs={t: x})


# 计算f(x)的导数
def cal_fx_diff(x):
    return f_diff().evalf(subs={t: x})


# 计算φ(x)
def phi(x):
    return x - cal_fx(x)


# 计算φ(x)的导数
def phi_diff(x):
    return 1 - cal_fx_diff(x)


# 符号函数，返回输入的正负
def sgn(x):
    if x >= 0:
        return 1
    else:
        return -1


# 简单迭代法，采用松弛加速
def iter(x, epsilon, k):
    x0 = x
    for i in range(k):
        temp = x
        x = (phi(x) - phi_diff(x0) * x) / (1 - phi_diff(x0))
        error = math.fabs(x - temp)
        if error < epsilon:
            return [x, error, i + 1]


# 牛顿法
def Newton(x, epsilon, k):
    for i in range(k):
        temp = x
        x = x - cal_fx(x) / cal_fx_diff(x)
        error = math.fabs(x - temp)
        if error < epsilon:
            return [x, error, i + 1]


# 弦割法
def Secant(x0, x1, epsilon, k):
    for i in range(k):
        x2 = x1 - (x1 - x0) * (cal_fx(x1) / cal_fx(x0)) / (cal_fx(x1) / cal_fx(x0) - 1)
        error = math.fabs(x2 - x1)
        x1, x0 = x2, x1
        if error < epsilon:
            return [x2, error, i + 1]


if __name__ == "__main__":
    epsilon: float = 1e-8  # 误差
    k = 100  # 最大迭代次数
    a = -1  # 区间左端点
    b = 5  # 区间右端点

    # 二分法
    while b - a >= 1:
        mid = (a + b) / 2
        if sgn(cal_fx(mid)) * sgn(cal_fx(a)) == -1:
            b = mid
        else:
            a = mid

    x0 = (a + b) / 2

    print("经过二分法，区间为[%s, %s]\n" % (a, b))

    [x, error, times] = iter(x0, epsilon, k)
    print("采用简单迭代法求解非线性方程：")
    print("x = %s" % x)
    print("误差：%s" % error)
    print("迭代次数：%s" % times)

    [x, error, times] = Newton(x0, epsilon, k)
    print("\n采用牛顿法求解非线性方程：")
    print("x = %s" % x)
    print("误差：%s" % error)
    print("迭代次数：%s" % times)

    [x, error, times] = Secant(b, a, epsilon, k)
    print("\n采用弦割法求解非线性方程：")
    print("x = %s" % x)
    print("误差：%s" % error)
    print("迭代次数：%s" % times)
