# -*-coding:utf-8 -*-
# 目标求解2*sin(x)+cos(x)最大值
import random
import math
import matplotlib.pyplot as plt
import numpy as np


def gene_encode(data, _interval, length, vars=3):
    """
    对点坐标进行二进制编码，转换为“基因”

    :param data:
    :param _interval:
    :param length:
    :param vars:
    :return: 点的坐标编码，{list:length*vars}
    """
    _gene = []
    for n in range(vars):
        num = int((data[n]) // _interval[n])
        for i in range(length):
            a = int(num // (2 ** (length - i - 1)) >= 1)
            b = num % (2 ** (length - i - 1))
            _gene.append(a)
            num = b
    return _gene


def gene_decode(gene_code, _interval, length=14, vars=3):
    """
    点坐标解码，将“基因”转变为坐标点

    :param gene_code:
    :param _interval:
    :param length:
    :param vars:
    :return: 点的实际坐标值，{list:vars}
    """
    point = []
    for n in range(vars):
        num = 0
        for i in range(length):
            num += gene_code[i + n * length] * (2 ** (length - i - 1))
        num = num * _interval[n]
        point.append(num)
    return point


def original_series(population, values_max, vars=3):
    """
    产生初始种群

    :param population:
    :param values_max:
    :param vars:
    :return: 初始种群，个体为不同坐标点，{list:population({list:vars})}
    """
    group = []
    for n in range(population):
        unit = []
        for i in range(vars):
            r = random.uniform(0, values_max[i])
            unit.append(round(r, 4))
        group.append(unit)
    return group


def eval_func(point):
    """
    计算适应度函数值和目标函数值

    :param point:
    :return: 适应度函数值，目标函数值
    """
    x1, x2, x3 = point
    f = 1000 - x1 ** 2 - 2 * x2 ** 2 - x3 ** 2 - x1 * x2 - x1 * x3

    h1 = x1 ** 2 + x2 ** 2 + x3 ** 2 - 25
    h2 = 8 * x1 + 14 * x2 + 7 * x3 - 56
    fit = 1081 - f - np.absolute(h1) - np.absolute(h2)
    return fit, f


def selection(fit_group, gene_group):
    """
    种群筛选，筛选出适应度高的个体，以便进行下一步的交叉

    :param fit_group:
    :param gene_group:
    :return:筛选出的适应度高的个体，个体为不同基因片段，{list:popuplation}
    """
    normal_fit_group, new_group = [], []
    fit, alive_rate = 0, 0
    alive_num = 200
    _sum = sum(fit_group)
    for i in range(len(fit_group)):
        fit += fit_group[i] / _sum
        normal_fit_group.append(fit)
    for i in range(alive_num):
        rate = random.uniform(0, 1)
        _index = 0
        while _index < len(gene_group):
            if rate < normal_fit_group[_index]:
                new_group.append(gene_group[_index])
                break
            else:
                _index += 1
    return new_group


def cross(gene_group):
    """
    单点交叉函数

    :param gene_group:
    :return: 交叉后的新种群，个体为不同基因片段，{list:popuplation}
    """
    random.shuffle(gene_group)
    gene_group1 = gene_group[:40]
    gene_group = gene_group[40:]
    for i in range(len(gene_group) // 2):
        temporary = []
        a = random.randint(0, len(gene_group[0]))
        temporary.extend(gene_group[i * 2][a:])
        gene_group[i * 2][a:] = gene_group[i * 2 + 1][a:]
        gene_group[i * 2 + 1][a:] = temporary
    gene_group.extend(gene_group1)
    return gene_group


def fit_judge(new_gene_group, _interval, best_point, best_fit, best_obj):
    new_fit, new_obj, new_point = [], [], []
    _err = 0
    for i in range(len(new_gene_group)):
        new_point.append(gene_decode(new_gene_group[i], _interval))
        _fit = eval_func(new_point[i])
        new_fit.append(_fit[0])
        new_obj.append(_fit[1])
    max_fit = max(new_fit)
    min_obj = new_obj[new_fit.index(max_fit)]
    if max_fit > best_fit:
        best_point = new_point[new_fit.index(max_fit)]
        best_fit = max_fit
        best_obj = min_obj
    return best_point, best_fit, new_fit, best_obj


def gene_mutate(gene_group, mutate_rate=0.001):
    for i in range(int(len(gene_group) * len(gene_group[0]) * mutate_rate)):
        series_num = random.randint(0, len(gene_group)-1)
        gene_num = random.randint(0, len(gene_group[0])-1)
        gene_group[series_num][gene_num] = int(not gene_group[series_num][gene_num])
    return gene_group


if __name__ == '__main__':
    max_value = [5, 4, 5]
    min_value = [0, 0, 0]
    gene_length = 14
    best_epoch = 0
    epoch = 1000
    interval = []
    for ii in range(len(max_value)):
        interval.append((max_value[ii] - min_value[ii]) / (2 ** gene_length - 1))
    origin_group = original_series(200, max_value, 3)
    origin_gene_group, fitness, objective = [], [], []
    for m in range(len(origin_group)):
        origin_gene_group.append(gene_encode(origin_group[m], interval, gene_length))
        fit = eval_func(origin_group[m])
        fitness.append(fit[0])  # 适应度函数值
        objective.append(fit[1])  # 目标函数值
    final_fit = max(fitness)
    final_obj = objective[fitness.index(final_fit)]
    final_point = origin_group[fitness.index(final_fit)]
    for index in range(epoch):
        selected_group = selection(fitness, origin_gene_group)
        crossed_group = cross(selected_group)
        crossed_group = gene_mutate(crossed_group)
        final_point, new_fit, fitness, final_obj = fit_judge(crossed_group, interval, final_point, final_fit, final_obj)
        print(index)
        if new_fit == final_fit:
            best_epoch += 1
        else:
            best_epoch = 0
        final_fit = new_fit
        if best_epoch >= 20:
            break
    print(final_fit, final_point, final_obj)

    # for i in range(epoch):
    # print(origin_gene_group)
