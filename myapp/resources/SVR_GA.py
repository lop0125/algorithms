# _*_ coding: utf-8 _*_
__author__ = "xy"

# _*_ coding: utf-8 _*_
__author__ = "xy"
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, basinhopping
from sklearn.svm import SVR
import myapp.dataDeal.dealUtil as du

def getEncodedLength(delta=[], boundaryList =[]):
    """
    得到染色体的长度，几个未知变量就有几个长度，染色体长度就是几个变量的长度相加
    :param delta: 解的精度
    :param boundaryList: 决策变量的上下界
    :return: 编码长度，它是一个tuple
    """
    lengths = []
    for k, i in enumerate(boundaryList):
        lower = i [0]
        upper = i [1]
        res = fsolve(lambda x: ((upper - lower) * 1 / delta[k]) - 2 ** x - 1, 50)
        length = math.ceil(np.floor(res[0])) # 取整
        lengths .append(length)
    return lengths

def getIntialPopulation(encodelength, populationSize):
    """
    初始化种群染色体
    :param encodelength:  个体染色体长度
    :param populationSize: 种群大小
    :return: 返回种群染色体
    """
    # 随机化初始化种群为0
    # 矩阵行数是种群的大小， 列数是每个个体的染色体长度
    chromosomes = np.zeros((populationSize, sum(encodelength)), dtype=np.uint8)
    for i in range(populationSize):
        # 随机创建0，1长度是染色体长度的大小
        chromosomes[i,:] = np.random.randint(0, 2, sum(encodelength))
    return chromosomes

def decodedChromosome(encodelength, chromosomes, boundarylist,delta):
    """
    得到基因的表现型
    :param encodelength: 染色体长度
    :param chromosomes: 种群
    :param boundarylist: 决策变量
    :return:
    """
    population = chromosomes.shape[0] # 得到人口大小
    variable = len(encodelength) # 得到决策变量的个数
    decodeValues = np.zeros((population, variable)) # 初始化解码值
    for k , chromosome in enumerate(chromosomes):
        chromosome = chromosome.tolist()
        start = 0
        for index, length in enumerate(encodelength):
            power = length - 1
            demical = 0
            for i in range(start, start + length):
                demical += chromosome[i] * (2 ** power)
                power -= 1
            lower = boundarylist[index][0]
            upper = boundarylist[index][1]
            digits = int(math.ceil(abs(np.log10(delta[index])))) # 精度的位数
            decodevalue = round((lower + demical * (upper - lower) / (2 ** length - 1)),
                                digits  )  # 想象一条直线方程，y= ax +b 求解
            decodeValues[k, index] = decodevalue
            start = length
    return decodeValues

def getFitnessValue (chromesomesdecoded,X, y):
    """
     计算适应度，这里适应度就是用百分比误差进行比较
    :param chromesomesdecoded:
    :param X: 测试集
    :param y:
    :return:
    """
    population, nums = chromesomesdecoded.shape
    # 初始化种群适应度为0
    fitnessValues= np.zeros((population,1))
    for i in range(population):
        # 计算每个染色体的适应度
        # for k in range(chromesomesdecoded[i, :]):
        clf = SVR(C=chromesomesdecoded[i,0],gamma=chromesomesdecoded[i,1])
        clf.fit(X,y)
        predict_value = clf.predict(X)
        fitnessValues[i, 0] = du.absolutePercentError(predict_value, y)  # 采用绝对百分比误差

    probability = fitnessValues / np.sum(fitnessValues)
    # 得到每个染色体被选中的累积概率
    cum_probability = np.cumsum(probability)
    return fitnessValues, cum_probability


# 新种群选择
def selectNewPopulation(chromosomes, cum_probability):
    """
    种群选择,个体染色体累积概率要大于随机的概率
    :param chromosomes:
    :param cum_probability:
    :return:
    """
    m, n = chromosomes.shape
    newpopulation = np.zeros((m,n))
    randoms = np.random.rand(m)
    for i ,random in enumerate (randoms):
        logical = cum_probability >= random
        index = np.where(logical==1)
        newpopulation [i, :] = chromosomes[index[0][0],:]

    return newpopulation


def crossover (population, Pc = 0.8):
    """
    交叉
    :param population:
    :param Pc: 交叉率
    :return:
    """
    m,n = population.shape
    numbers = np.uint8(m * Pc) # 根据交叉率得到交叉的个数
    if numbers % 2 != 0:
        numbers += 1
    updatepopulation = np.zeros((m,n),dtype=np.uint8)
    indexs = random.sample(range(m), numbers) # 从m 中随机获取numbers 个数，并返回,indexs 是数组
    # 不交叉的片段进行复制
    for i in range(m):
        if not indexs.__contains__(i):
            updatepopulation[i, :] = population[i,:]

    # 交叉

    while len(indexs) > 0:
        # 从元组的最后面两个数开始计算
        a = indexs.pop()
        b = indexs.pop()
        crossoverPoint = random.sample(range(1,n), 1)
        crossoverPoint = crossoverPoint[0]
        updatepopulation[a, 0:crossoverPoint] = population[a, 0:crossoverPoint]
        updatepopulation[a, crossoverPoint:] = population[b,crossoverPoint:]
        updatepopulation[b, 0:crossoverPoint] = population[b, 0:crossoverPoint]
        updatepopulation[b, crossoverPoint]= population[a, crossoverPoint]
    return updatepopulation
    pass

def mutation(population, pm= 0.01):
    """
    基因变异
    :param population:
    :param pm:
    :return:
    """
    m, n = population.shape
    updatepopulation = np.copy(population)
    numbers = np.uint8(m * n * pm) #变异个数
    mutationGeneIndex = random.sample(range(0,m * n), numbers)

    for gene in mutationGeneIndex:
        # 确定是第几个染色体
        chromesomeIndex = gene // n  # // 是整数除法
        # 确定是第几个基因位
        geneIndex = gene % n
        if updatepopulation[chromesomeIndex, geneIndex] == 0:
            updatepopulation[chromesomeIndex, geneIndex] = 1
        else:
            updatepopulation[chromesomeIndex, geneIndex] = 0
    return updatepopulation
    pass
# def test (upper,lower,delta=0.01):
#     res = fsolve(lambda x: ((upper - lower) * 1 / delta) - 2 ** x - 1, 50)
#     print(type(res[0]))

# 适应度函数,就是SVR 模型方程
# def fitnessFunction(chromesomesdecoded):
#
#     SVR(C=chromesomesdecoded)


def main(args):
    data = args['data']
    data = eval(data)
    source = data['value']
    train_data, test= du.isolation_data(source)  # 将数据集分为训练集跟测试集
    # 将数据集分为训练集，样本集
    dataset,train_X, label_y = du.create_dataset(train_data)
    max_iter = 1000
    # n_samples, n_features = 50, 5
    # # np.random.seed(0) # 随机数的种子保证每次随机数的大小相同
    # y = np.random.random_sample(n_samples)
    # X = np.random.random_sample([n_samples, n_features])
    # print(y, X)
    errors = []
    optimalSolutions =[]
    # 决策变量取值范围
    decisionVariables = [[1, 1000],[1e-2,0.1]]
    deltaVariables = [0.1, 0.001]
    lengthdecode = getEncodedLength(delta=deltaVariables , boundaryList=decisionVariables)
    chromesomeCode = getIntialPopulation(lengthdecode, 50)
    for iteration in range(max_iter):  # 种群迭代
       # 染色体解码
        decoded = decodedChromosome(lengthdecode,chromesomeCode,decisionVariables,deltaVariables)
        evalvalue , cum_probility = getFitnessValue(decoded,train_X,label_y)
        # 选择新的种群
        newpopulation= selectNewPopulation(chromesomeCode,cum_probility)
        crossoverpopulation = crossover(newpopulation)

        mutationpopulation = mutation(crossoverpopulation)

        final_decoded = decodedChromosome(lengthdecode,mutationpopulation,decisionVariables,deltaVariables)
        final_fitnessvalue, final_cum_probility = getFitnessValue(final_decoded,train_X,label_y)
        errors.append(np.min(list(final_fitnessvalue)))
        index = np.where(final_fitnessvalue == min(list(final_fitnessvalue)))
        optimalSolutions.append(final_decoded[index[0][0],:])
        chromesomeCode = mutationpopulation

    error = np.min(errors)
    optimalIndex = np.where(errors == error)
    optimalSolution = optimalSolutions[optimalIndex[0][0]]
    clf = SVR (C=optimalSolution[0], gamma=optimalSolution[1])
    clf.fit(train_X,label_y)
    optimalValue = clf.predict(train_X)
    abs_error = du.absoluteError(optimalValue, label_y)
    plt.plot(range(len(label_y)),label_y,'r')
    plt.plot(range(len(optimalValue)),optimalValue,'b')
    plt.show()
    print(abs_error, error)
    return {
             "optimalSolution":optimalSolution.tolist(),
             "optimalValue":optimalValue.tolist()
    }


if __name__ == '__main__':
    solution ,value= main()
    print(solution, value)
    pass