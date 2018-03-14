from numpy import *
from math import log

from machine_learning.treePlotter import getTreePlotter

"""
决策树模型(ID3算法)
"""
def createDataSet():
    """
    创建数据集
    """
    dataSet = [
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算数据集的香农熵
    具体过程 计算每个分类在当前数据集中的比例props(选择当前分类的概率)
    根据公式，使 -props * log2(props) 求和
    其中-log2(props)为单个信息量
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
条件熵 在已经一个特征的值的情况下该数据集的香农熵 那么这个用哪个值固定特征呢？每个值都要用到并且取均值
信息增益 数据集熵-条件熵  H(X,A) = H(X) - ∑pi * H(pi) 其中pi为某一确定值的属性概率
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
    :param charIndex:维度
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
    print(classList)
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


def classify(tree, labels, vector):
    """
    使用决策树对vector进行分类
    :param tree: 构建好的决策树
    :param labels: 特征值
    :param vector: 待测向量
    """
    firstFeatStr = list(tree.keys())[0]
    secondDic = tree[firstFeatStr]
    if type(secondDic).__name__ != 'dict':
        return str(secondDic)
    else:
        featIndex = labels.index(firstFeatStr)
        for key in list(secondDic.keys()):
            if vector[featIndex] == key:
                if type(secondDic[key]).__name__ == 'dict':
                    return classify(secondDic, labels, vector)
                else:
                    return secondDic[key]


def storeTree(tree, filename):
    """
    存入决策树
    :param tree:
    :param filename:
    :return:
    """
    import pickle
    with open(filename, "wb") as fw:
        pickle.dump(tree, fw)
        # fw.write("好的")


def getTree(filename):
    """
    取出决策树
    :param filename:
    :return:
    """
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)


def fileToDataSet(filename):
    with open(filename) as f:
        array = [line.strip().split("\t") for line in f.readlines()]
        return array


if __name__ == '__main__':
    # data = createDataSet()
    # dataSet = data[0]
    # labels = ['alive without water', 'jiaopu']
    dataSet = fileToDataSet("F:\pycharm_workapace\数据样本\Ch03\lenses.txt")
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    tree = createTree(dataSet, labels[:])
    getTreePlotter.createPlot(tree)  # 画出决策树
    # result = classify(tree, labels[:], [1, 0])  # 测试决策树
    # print(result)
    storeTree(tree, "f:/python_treeplotter.txt")  # 存储决策树
    # print(getTree("f:/python_treeplotter.txt"))  # 取出决策树
