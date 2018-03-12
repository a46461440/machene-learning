from matplotlib import pyplot
from numpy import *


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
            resultMat[index, :] = lineArray[0:3]
            classLabelVector.append(int(lineArray[-1]))
            index += 1
        return resultMat, classLabelVector


data = file2matrix("F:\kNN测试数据.txt", 3)
figure = pyplot.figure()
ax = figure.add_subplot(111)
# ax.scatter(data[0][:, 1], data[0][:, 2], 15 * array(data[1]), 15 * array(data[1]))
# ax.scatter(data[0][:, 1], data[0][:, 2])
for i in range(len(data[0])):
    if data[1][i] > 5:
        ax.scatter(data[0][i:i+1, 1], data[0][i:i+1, 2], c='red', marker='o', s=30, label='big')
    else:
        ax.scatter(data[0][i:i+1, 1], data[0][i:i+1, 2], c='blue', marker='x', s=30, label='small')
pyplot.title("test")
pyplot.xlabel("x")
pyplot.ylabel("y")
pyplot.show()
