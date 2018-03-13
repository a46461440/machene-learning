from numpy import *
from math import log

"""
决策树模型
"""


def createDataSet():
    """
    创建数据集
    """
    dataSet = [
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算数据集的香农熵
    :param dataSet:
    """
    dataSetSize = len(dataSet)
    labelCount = {}
    shannonEnt = 0.00
    for vector in dataSet:
        char = str(vector[-1])
        if char not in labelCount.keys():
            labelCount[char] = 0
        labelCount[char] += 1
    for key in labelCount:
        prob = float(labelCount[key]) / dataSetSize
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


"""
条件熵 在已经一个特征的值的情况下该数据集的香农熵
"""


def spilitDataSet(dataSet, charIndex, value):
    """
    1.划分数据集
    :param dataSet:
    :param charIndex:
    :param value:
    """
    resetDataSet = []
    for vector in dataSet:
        if vector[charIndex] == value:
            reductVector = vector[:charIndex]
            reductVector.extend(vector[charIndex + 1:])
            resetDataSet.append(reductVector)
    return resetDataSet


def calcConditionalEntropy(dataSet, charIndex, uniqueValues):
    """
    2.计算条件熵
    :param dataSet:数据集
    :param i:维度
    :param uniqueValues:数据集特征集合
    :return:
    """
    conditionEnt = 0.0
    for char in uniqueValues:
        subDataSet = spilitDataSet(dataSet, charIndex, char)
        prop = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率 (这样可以保证所欲特征值的维度都是一样的)
        conditionEnt += prop * calcShannonEnt(subDataSet)
    return conditionEnt


def chooseBestFeatureToSpilit(dataSet):
    """
    选择最好的数据划分方式(选择最好的特征) 信息增益
    :param dataSet:
    """
    charnum = len(dataSet[0]) - 1
    baseShannoEnt = calcShannonEnt(dataSet)
    bestShannoEnt = 0.0
    bestCharIndex = -1
    for charIndex in range(charnum):
        featList = [example[charIndex] for example in dataSet]
        uniqueValues = set(featList)
        currCharShannoEnt = baseShannoEnt - calcConditionalEntropy(dataSet, charIndex, uniqueValues)
        if currCharShannoEnt > bestShannoEnt:
            bestShannoEnt = currCharShannoEnt
            bestCharIndex = charIndex
    return bestCharIndex


def majorityCnt(classList):
    classCount = {}
    for clazz in classList:
        if clazz not in classCount:
            classCount[clazz] = 0
        classCount[clazz] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1])
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet:
    :param labels:所有特征的标签
    :return:
    """
    classList = [example[-1] for example in dataSet]  # 取出所有的分类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeatureIndex = chooseBestFeatureToSpilit(dataSet)
    bestFeature = labels[bestFeatureIndex]
    tree = {bestFeature: {}}
    del (labels[bestFeatureIndex])
    features = [example[bestFeatureIndex] for example in dataSet]
    uniqueFeature = set(features)
    for value in uniqueFeature:
        newLabels = labels[:]  # 防止因python传址导致的问题
        tree[bestFeature][value] = createTree(spilitDataSet(dataSet, bestFeatureIndex, value), newLabels)
    return tree


if __name__ == '__main__':
    data = createDataSet()
    dataSet = data[0]
    labels = ['alive without water', 'jiaopu']
    tree = createTree(dataSet, labels)
    print(tree)
    # bestCharIndex = chooseBestFeatureToSpilit(dataSet)
    # print(bestCharIndex)
