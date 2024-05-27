import sympy as sp
import sympy.integrals.trigonometry
import numpy as np
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt
# 计算信息挖掘值yi when li == 1，求积分函数代号为yi1
x_yi1 = sp.Symbol('x_yi1')
mu = 1.0168
sigma2 = 0.0001942
fx_QiuJiFen_yi1 = x_yi1 * (sp.exp((-(x_yi1 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
resultUp_yi1 = sp.integrate(fx_QiuJiFen_yi1, (x_yi1, 1.020, float("inf")))
resultDown_yi1 = normal_distribution((1.020 - mu) / sp.sqrt(sigma2))
print(float(resultUp_yi1) / resultDown_yi1)


# 计算信息挖掘值yi when li == 0，求积分函数代号为yi2
x_yi2 = sp.Symbol('x_yi2')
mu = 1.0168
sigma2 = 0.0001942
fx_QiuJiFen_yi2 = x_yi2 * (sp.exp((-(x_yi2 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
resultUp_yi2 = sp.integrate(fx_QiuJiFen_yi2, (x_yi2, 1.020, float("inf")))
resultDown_yi2 = 1 - normal_distribution((1.020 - mu) / sp.sqrt(sigma2))
print(float(resultUp_yi2) / resultDown_yi2)

# 计算信息挖掘值ti when li == 1，求积分函数代号为ti1
x_ti1 = sp.Symbol('x_ti1')
mu = 1.0168
sigma2 = 0.0001942
fx_QiuJiFen_ti1 = (x_ti1 ** 2) * (sp.exp((-(x_ti1 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
resultUp_ti1 = sp.integrate(fx_QiuJiFen_ti1, (x_ti1, 1.020, float("inf")))
resultDown_ti1 = normal_distribution((1.020 - mu) / sp.sqrt(sigma2))
print(float(resultUp_ti1) / resultDown_ti1)

# 计算信息挖掘值ti when li == 0，求积分函数代号为ti2
x_ti2 = sp.Symbol('x_ti2')
mu = 1.0168
sigma2 = 0.0001942
fx_QiuJiFen_ti2 = (x_ti2 ** 2) * (sp.exp((-(x_ti2 - mu) ** 2) / (2 * sigma2)) / (sp.sqrt(2 * sp.pi) * sp.sqrt(sigma2)))
resultUp_ti2 = sp.integrate(fx_QiuJiFen_ti2, (x_ti2, 1.020, float("inf")))
resultDown_ti2 = 1 - normal_distribution((1.020 - mu) / sp.sqrt(sigma2))
print(float(resultUp_ti2) / resultDown_ti2)