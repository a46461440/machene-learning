from numpy import *
from math import log


def calcShannoEnt(dataSet):
    """
    计算数据集香农熵
    """
    size = len(dataSet)
    classCount = {}
    for vector in dataSet:
        clazz = vector[-1]
        if clazz not in list(classCount.keys()):
            classCount[clazz] = 0
        classCount[clazz] += 1
    shannoEnt = 0.0
    for key in classCount:
        probs = classCount[key] / float(size)
        shannoEnt -= probs * log(probs, 2)
    return shannoEnt


def chooseBestInfomationGain(dataSet):
    baseShannoEnt = calcShannoEnt(dataSet)
    charnum = len(dataSet) - 1