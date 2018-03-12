from Tools.scripts.treesync import raw_input
from numpy import tile, mat, array, zeros


# intX为目标向量
# dataSet为训练集
# labels为训练集结果
# k为k邻近值
def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMats = tile(intX, [dataSetSize, 1]) - dataSet
    distance = ((diffMats ** 2).sum(axis=1)) ** 0.5
    sortedIndex = distance.argsort()
    classCount = {}
    for i in range(k):
        target = labels[sortedIndex[i]]
        classCount[target] = classCount.get(target, 0) + 1
    resultIndex = sorted(classCount.items(), key=lambda item: item[1])
    return resultIndex[0][0]


# 归一化数据
def autoNorm(dataSet):
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    gap = maxValue - minValue
    size = dataSet.shape[0]
    return minValue, gap, (dataSet - tile(minValue, [size, 1])) / tile(gap, [size, 1])


# 解析文件
def file2matrix(filename, charnum):
    with open(filename) as fr:
        dataArray = fr.readlines()
        dataSize = len(dataArray)
        resultMat = zeros((dataSize, charnum))
        classLabelVector = []
        index = 0
        for line in dataArray:
            line = line.strip()
            lineArray = line.split("\t")
            resultMat[index, :] = lineArray[0:charnum]
            classLabelVector.append(lineArray[-1])
            index += 1
        return resultMat, classLabelVector


# 测试方法
# hoRatio为测试数据占样本数据比例
def datingClassTest(filename, charnum, hoRatio):
    dataSet, labels = file2matrix(filename, charnum)
    # 数据归一化 所有特性的重要性相同
    minValue, gap, dataSetFilter = autoNorm(dataSet)
    dataSize = dataSet.shape[0]
    testCaseNum = int(dataSize * hoRatio)
    errorCount = 0
    for i in range(testCaseNum):
        result = classify0((dataSetFilter[i, 0:charnum] - minValue) / gap, dataSetFilter[testCaseNum:, 0:charnum],
                           labels[charnum:], 5)
        if result != labels[i]: errorCount += 1
        print("the classifier came back with:%s, the real answer is:%s" % (result, labels[i]))
    print("the total error rate is:%f" % (errorCount / float(dataSize - errorCount)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    ffMiles = float(raw_input('frequent flier miles earned per year'))
    percentTats = float(raw_input('percentage of time spent playing video games'))
    iceCream = float(raw_input('liters of ice cream consumed per year'))
    intX = array([ffMiles, percentTats, iceCream])
    data = file2matrix('F:\pycharm_workapace\数据样本\Ch02\datingTestSet2.txt', 3)
    minValue, gap, dataSet = autoNorm(data[0])
    labels = data[1]
    resultIndex = classify0((intX - minValue) / gap, dataSet, labels, 3)
    print(resultIndex)
    print("You will probably like this person: %s" % (resultList[int(resultIndex) - 1]))


if __name__ == '__main__':
    # classifyPerson()
    datingClassTest('F:\pycharm_workapace\数据样本\Ch02\datingTestSet2.txt', 3, 0.09)