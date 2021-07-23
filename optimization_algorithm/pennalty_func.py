# -*-coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from sympy import symbols, diff, zeros, solve
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


def obj_func():
    """
    定义优化模型
    :return:优化模型的目标函数、不等式约束、等式约束、设计变量
    """
    # 定义设计变量
    _x1 = symbols("x1")
    _x2 = symbols("x2")
    _x3 = symbols("x3")
    # 定义目标函数
    func = 1000 - _x1 ** 2 - 2 * _x2 ** 2 - _x3 ** 2 - _x1 * _x2 - _x1 * _x3
    # 定义不等式约束
    g1 = -_x1
    g2 = -_x2
    g3 = -_x3
    # 定义等式约束
    h1 = _x1 ** 2 + _x2 ** 2 + _x3 ** 2 - 25
    h2 = 8 * _x1 + 14 * _x2 + 7 * _x3 - 56
    return func, (g1, g2, g3), (h1, h2), [_x1, _x2, _x3]


def calc_func_values(obj_f, obj_g, obj_h, vars_list, X):
    """
    计算模型的目标函数值、不等式约束值、等式约束值
    :param obj_f:目标函数
    :param obj_g:不等式约束
    :param obj_h:等式约束
    :param vars_list:设计变量表
    :param X:设计方案点
    :return:模型的目标函数值、不等式约束值、等式约束值
    """
    obj_h = list(obj_h)
    obj_g = list(obj_g)
    # 计算目标函数值
    for i in range(len(vars_list)):
        obj_f = obj_f.subs(vars_list[i], X[i])
    # 计算不等式约束值
    for m in range(len(obj_g)):
        for i in range(len(vars_list)):
            obj_g[m] = list(obj_g)[m].subs(vars_list[i], X[i])
    # 计算等式约束值
    for m in range(len(obj_h)):
        for i in range(len(vars_list)):
            obj_h[m] = obj_h[m].subs(vars_list[i], X[i])
    return obj_f, obj_g, obj_h


def conjugate_gradient(X_init, obj, vars_name):
    """
    计算无约束优化模型最优解的共轭梯度法
    :param X_init: 初始点
    :param obj: 目标函数
    :param vars_name: 设计变量表
    :return: 无约束优化模型最优解
    """
    a = symbols("a")
    vars_count = len(vars_name)
    gradient = np.array(list(zeros(1, 3)))
    f0 = obj
    for index in range(vars_count):
        gradient[index] = diff(obj, vars_name[index])
        f0 = f0.subs(vars_name[index], X_init[index])
    min_f = f0  # 定义最优解变量min_f
    # 计算初始方向向量，为负梯度方向
    d0 = -gradient  # 定义初始搜索方向d0
    for index in range(vars_count):
        for i in range(vars_count):
            d0[index] = d0[index].subs(vars_name[i], X_init[i])
        d0[index] = float(d0[index])
    # 开始迭代
    for num in range(100):
        X1 = X_init + a * d0
        f1 = obj
        # 计算最优步长
        for i in range(len(X1)):
            f1 = f1.subs(vars_name[i], X1[i])  # 生成新的目标函数f1(a)
        solution = solve(diff(f1, a), a)  # 求解该目标函数的极值点a
        best_a = float(solution[0])  # 格式转换
        # 产生新的迭代点
        for iii in range(len(X1)):
            X1[iii] = X1[iii].subs(a, best_a)
        # 由共轭梯度法产生新的迭代方向
        d1 = -gradient
        for index in range(vars_count):
            for i in range(vars_count):
                d1[index] = d1[index].subs(vars_name[i], X1[i])
            d1[index] = float(d1[index])
        b = np.linalg.norm(d1) ** 2 / np.linalg.norm(d0)
        d0 = d1 + b * d0  # 计算共轭方向
        X_init = X1  # 将上一次的迭代终点赋值给迭代初始点
        f1 = f1.subs(a, best_a)
        # 迭代终止条件：目标函数下降量足够小
        if np.abs(min_f - f1) < 0.0001:
            break
        min_f = f1
    for i in range(len(X_init)):
        X_init[i] = float(X_init[i])  # 转换格式
    return X_init


def pennalty_func(obj_f, obj_g, obj_h, vars_list, X0):
    """
    计算有约束优化模型的外点惩罚函数法
    :param obj_f:目标函数
    :param obj_g:不等式约束
    :param obj_h:等式约束
    :param vars_list:设计变量表
    :param X0:设计方案点
    :return:
    """
    X_list, best_list = [], []
    r = 100000000  # 初始惩罚因子
    values = calc_func_values(obj_f, obj_g, obj_h, vars_list, X0)  # 计算目标函数值
    best_values = values[0]  # 将初始点目标函数值赋值给最优解变量
    X_list.append(X0)
    best_list.append(best_values)
    print(X0, best_values)
    for m in range(100):
        # 建立新的带有惩罚项的惩罚函数
        func_fy = obj_f
        for i in range(len(obj_g)):
            if values[1][i] > 0:
                func_fy += r * obj_g[i] ** 2
        for i in range(len(obj_h)):
            func_fy += r * obj_h[i] ** 2
        # 利用共轭梯度法计算当次迭代的近似最优解
        X1 = conjugate_gradient(X0, func_fy, vars_list)
        values = calc_func_values(obj_f, obj_g, obj_h, vars_list, X1)
        # 迭代终止条件：目标函数下降量足够小
        if np.abs(values[0] - best_values) < 0.000001:
            X0 = X1
            best_values = values[0]
            X_list.append(X0)
            best_list.append(best_values)
            break
        else:
            X0 = X1
            best_values = values[0]
            X_list.append(X0)
            best_list.append(best_values)
        # r = 10 * r

        print(X0, best_values)
    print(X0, best_values)
    return X_list, best_list


if __name__ == '__main__':
    x0 = [2, 1, 2]
    f = obj_func()
    X, F = pennalty_func(f[0], f[1], f[2], f[3], x0)
    X = array(X)

    plt.plot(X[:, 0])
    plt.plot(X[:, 1])
    plt.plot(X[:, 2])
    plt.legend(['x1', 'x2', 'x3'])
    plt.xlabel('迭代次数', FontProperties=font)
    plt.ylabel('设计变量值', FontProperties=font)
    plt.title('外点惩罚函数法迭代中设计变量变化过程', FontProperties=font)
    plt.show()

    plt.plot(F[:])
    plt.legend(['f(X)'])
    plt.xlabel('迭代次数', FontProperties=font)
    plt.ylabel('目标函数值', FontProperties=font)
    plt.title('外点惩罚函数法迭代中目标函数值变化过程', FontProperties=font)
    plt.show()

    # print(m)
