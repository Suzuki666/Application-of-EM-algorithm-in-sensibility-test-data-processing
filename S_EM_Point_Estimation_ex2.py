# 2023/04/10
# Zhongke Sun
import sympy as sp
import sympy.integrals.trigonometry
import numpy as np
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt


# Calculate the distributin function of standard normal distribution 计算标准正态分布的分布函数
def normal_distribution(x):  # Operate x<0 condition (target point is in the left side of the distribution center)处理x<0(目标点在分布中心左侧)的情况
    if x < 0:
        return 1 - normal_distribution(-x)
    if x == 0:
        return 0.5  # Solve the integral of the probability density of the standard normal distribution 求标准正态分布的概率密度的积分
    s = 1 / 10000
    xk = []
    for i in range(int(x / s)):
        integral = fx_normal_distribution((i + 1) * s)
        xk.append(integral)
    sum = 0
    for each in xk:
        sum += each
    return 0.5 + sum * s


def fx_normal_distribution(x):
    return math.exp((-(x) ** 2) / 2) / (math.sqrt(2 * math.pi))


# Set the initial value of the parameter mu and sigma2 设置参数mu和sigma2的初值
mu = 0
sigma2 = 0


# Calculate the information dig value yi when li == 1, calculate the integral function named yi1, pack function 计算信息挖掘值yi when li == 1，求积分函数代号为yi1，函数打包
def fx_JiFen_yi1(xmi, mu, sigma2):
    x_yi1 = sp.Symbol('x_yi1')
    fx_QiuJiFen_yi1 = x_yi1 * (sp.exp((-(x_yi1 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
    resultUp_yi1 = sp.integrate(fx_QiuJiFen_yi1, (x_yi1, -float("inf"), xmi))
    resultDown_yi1 = normal_distribution((xmi - mu) / sp.sqrt(sigma2))
    return float(resultUp_yi1) / resultDown_yi1


# Calculate the information dig value yi when li == 0, calculate the integral function named yi2, pack function计算信息挖掘值yi when li == 0，求积分函数代号为yi2，函数打包
def fx_JiFen_yi2(xmi, mu, sigma2):
    x_yi2 = sp.Symbol('x_yi2')
    fx_QiuJiFen_yi2 = x_yi2 * (sp.exp((-(x_yi2 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
    resultUp_yi2 = sp.integrate(fx_QiuJiFen_yi2, (x_yi2, xmi, float("inf")))
    resultDown_yi2 = 1 - normal_distribution((xmi - mu) / sp.sqrt(sigma2))
    return float(resultUp_yi2) / resultDown_yi2


# Calculate the information dig value ti when li == 1, calculate the integral function named ti1, pack function 计算信息挖掘值ti when li == 1，求积分函数代号为ti1，函数打包
def fx_JiFen_ti1(xmi, mu, sigma2):
    x_ti1 = sp.Symbol('x_ti1')
    fx_QiuJiFen_ti1 = (x_ti1 ** 2) * (
                sp.exp((-(x_ti1 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
    resultUp_ti1 = sp.integrate(fx_QiuJiFen_ti1, (x_ti1, -float("inf"), xmi))
    resultDown_ti1 = normal_distribution((xmi - mu) / sp.sqrt(sigma2))
    return float(resultUp_ti1) / resultDown_ti1


# Calculate the information dig value ti when li == 0, calculate the integral function named ti2, pack function 计算信息挖掘值ti when li == 0，求积分函数代号为ti2，函数打包
def fx_JiFen_ti2(xmi, mu, sigma2):
    x_ti2 = sp.Symbol('x_ti2')
    fx_QiuJiFen_ti2 = (x_ti2 ** 2) * (
                sp.exp((-(x_ti2 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
    resultUp_ti2 = sp.integrate(fx_QiuJiFen_ti2, (x_ti2, xmi, float("inf")))
    resultDown_ti2 = 1 - normal_distribution((xmi - mu) / sp.sqrt(sigma2))
    return float(resultUp_ti2) / resultDown_ti2


# print(fx_JiFen_yi1(xmi, mu, sigma2))
# print(fx_JiFen_yi2(xmi, mu, sigma2))
# print(fx_JiFen_ti1(xmi, mu, sigma2))
# print(fx_JiFen_ti2(xmi, mu, sigma2))

if __name__ == '__main__':
    # Input the sensitivity test result 导入感度试验结果
    workbook = pd.read_excel('安全解除保险距离试验结果.xlsx')
    XuHao = workbook["序号i"]
    Response = workbook["响应情况li"]
    TestCondition = workbook["试验水平xmi"]
    # Number 序号数目
    n = len(XuHao)
    # Initialize the information dig value array 初始化信息挖掘值数组
    Infor_yi = pd.Series(np.zeros(n, float))
    Infor_ti = pd.Series(np.zeros(n, float))
    # Print inputed data 打印出导入的数据
    print(workbook)
    # Specify normal distribution parameter, standard error = Sigma and average value = Mu指定正态分布参数，标准差为Sigma，均值为Mu
    Mu_old = 0
    Sigma2_old = 0
    count = 0
    # Calculate the initial value of Mu and Sigma2 计算Mu和Sigma2的初值
    Mu_new = sum(TestCondition) / n
    Sigma2_new = sum(pow((TestCondition - Mu_new), 2)) / (n - 1)
    print("Mu初值", Mu_new)
    print("Sigma2初值", Sigma2_new)
    # Big Iteration Loop 大的迭代循环
    while abs(Mu_new - Mu_old) + abs(Sigma2_new - Sigma2_old) > 0.000001 and count < 51:
        # Count Iteration 计数迭代次数
        count = count + 1
        Mu_old = Mu_new
        Sigma2_old = Sigma2_new
        # Small Loop 小的循环
        for i in range(n):
            if Response[i] == 1:
                # Calcualte the information dig value yi and ti when sucessful response 计算响应成功时的信息挖掘值yi和ti
                Infor_yi[i] = fx_JiFen_yi1(TestCondition[i], Mu_new, Sigma2_new)
                Infor_ti[i] = fx_JiFen_ti1(TestCondition[i], Mu_new, Sigma2_new)
            elif Response[i] == 0:
                # Calcualte the information dig value yi and ti when unsucessful response 计算未成功响应时的信息挖掘值yi和ti
                Infor_yi[i] = fx_JiFen_yi2(TestCondition[i], Mu_new, Sigma2_new)
                Infor_ti[i] = fx_JiFen_ti2(TestCondition[i], Mu_new, Sigma2_new)
        # Look at the information dig array for this round 看这轮迭代的信息挖掘数组
        print("信息挖掘值数组Infor_yi")
        print(Infor_yi)
        print("信息挖掘值数组Infor_ti")
        print(Infor_ti)
        # Calculate new parameter according to 2-7 根据式2-7计算新的参数
        Mu_new = sum(Infor_yi) / n
        Sigma2_new = (sum(Infor_ti) - pow(sum(Infor_yi), 2) / n) / n
        print("新的Mu：", Mu_new)
        print("新的Sigma2：", Sigma2_new)
        print("迭代次数：", count)
