# 迭代画图
# 2023/04/21
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 导入感度试验结果
    workbook = pd.read_excel('迭代画图.xlsx')
    Count = workbook["i"]
    Mu = workbook["mu"]
    Sigma2 = workbook["sigma2"]
    print(Mu)
    print(Sigma2)
    plt.plot(Count, Mu)
    plt.title("参数Mu迭代过程")
    plt.plot(Count, Sigma2)
    plt.title("参数Sigma2迭代过程")
