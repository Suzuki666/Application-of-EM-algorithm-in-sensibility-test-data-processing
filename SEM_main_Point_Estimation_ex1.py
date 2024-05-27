# 2023/04/06
# Zhongke Sun
import sympy as sp
import sympy.integrals.trigonometry
import numpy as np
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt


# Calculate the distribution function of standard normal distribution
def normal_distribution(x):  # Operate the condition that x<0(Target point in the left side of the distribution center)
    if x < 0:
        return 1 - normal_distribution(-x)
    if x == 0:
        return 0.5  # Calculate the integral of probability density of the standard normal distribution
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


# Set the initial value of parameter mu and sigma2
mu = 0
sigma2 = 0


# Calculate information dig value yi when li == 1，calculate integral function named yi, Pack the function
def fx_JiFen_yi1(xmi, mu, sigma2):
    x_yi1 = sp.Symbol('x_yi1')
    fx_QiuJiFen_yi1 = x_yi1 * (sp.exp((-(x_yi1 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
    resultUp_yi1 = sp.integrate(fx_QiuJiFen_yi1, (x_yi1, -float("inf"), xmi))
    resultDown_yi1 = normal_distribution((xmi - mu) / sp.sqrt(sigma2))
    return float(resultUp_yi1) / resultDown_yi1


# Calculate information dig value yi when li == 0，calculate integral function named yi2, Pack the function
def fx_JiFen_yi2(xmi, mu, sigma2):
    x_yi2 = sp.Symbol('x_yi2')
    fx_QiuJiFen_yi2 = x_yi2 * (sp.exp((-(x_yi2 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
    resultUp_yi2 = sp.integrate(fx_QiuJiFen_yi2, (x_yi2, xmi, float("inf")))
    resultDown_yi2 = 1 - normal_distribution((xmi - mu) / sp.sqrt(sigma2))
    return float(resultUp_yi2) / resultDown_yi2


# Calculate information dig value ti when li == 1，calculate integral function named ti1, Pack the function
def fx_JiFen_ti1(xmi, mu, sigma2):
    x_ti1 = sp.Symbol('x_ti1')
    fx_QiuJiFen_ti1 = (x_ti1 ** 2) * (
                sp.exp((-(x_ti1 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
    resultUp_ti1 = sp.integrate(fx_QiuJiFen_ti1, (x_ti1, -float("inf"), xmi))
    resultDown_ti1 = normal_distribution((xmi - mu) / sp.sqrt(sigma2))
    return float(resultUp_ti1) / resultDown_ti1


# Calculate information dig value ti when li == 0，calculate integral function named ti2, Pack the function
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
    # Input sensitivity test result
    workbook = pd.read_excel('发火电压感度试验结果.xlsx')
    XuHao = workbook["序号i"]
    Response = workbook["响应情况li"]
    TestCondition = workbook["试验水平xmi"]
    # Number 序号数目
    n = len(XuHao)
    # Initialize information dig value array 初始化信息挖掘值数组
    Infor_yi = pd.Series(np.zeros(n, float))
    Infor_ti = pd.Series(np.zeros(n, float))
    # print inputing data 打印出导入的数据
    print(workbook)
    # Specify normal distribution parameter, standard error is Sigma, Average value is Mu
    Mu_old = 0
    Sigma2_old = 0
    count = 0
    # Calculate the initial value of Mu and Sigma2
    Mu_new = sum(TestCondition) / n
    Sigma2_new = sum(pow((TestCondition - Mu_new), 2)) / (n - 1)
    print("Mu初值", Mu_new)
    print("Sigma2初值", Sigma2_new)
    # Big Iteration Loop大的迭代循环
    while abs(Mu_new - Mu_old) + abs(Sigma2_new - Sigma2_old) > 0.000001:
        # Count Iteration number 计数迭代次数
        count = count + 1
        Mu_old = Mu_new
        Sigma2_old = Sigma2_new
        # Small Loop 小的循环
        for i in range(n):
            if Response[i] == 1:
                # Calculate Information Dig value yi and ti when sucessful response计算响应成功时的信息挖掘值yi和ti
                Infor_yi[i] = fx_JiFen_yi1(TestCondition[i], Mu_new, Sigma2_new)
                Infor_ti[i] = fx_JiFen_ti1(TestCondition[i], Mu_new, Sigma2_new)
            elif Response[i] == 0:
                # Calculate Information Dig value yi and ti when unsucessful response计算未响应成功时的信息挖掘值yi和ti
                Infor_yi[i] = fx_JiFen_yi2(TestCondition[i], Mu_new, Sigma2_new)
                Infor_ti[i] = fx_JiFen_ti2(TestCondition[i], Mu_new, Sigma2_new)
        # Look at the Information dig array for this round 看这轮迭代的信息挖掘数组
        print("信息挖掘值数组Infor_yi")
        print(Infor_yi)
        print("信息挖掘值数组Infor_ti")
        print(Infor_ti)
        # Calculate new parameter according to formula 2-7
        Mu_new = sum(Infor_yi) / n
        Sigma2_new = (sum(Infor_ti) - pow(sum(Infor_yi), 2) / n) / n
        print("新的Mu：", Mu_new)
        print("新的Sigma2：", Sigma2_new)
        print("迭代次数：", count)
