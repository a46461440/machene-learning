# numpy测试函数 第一次


# tile函数 将一个数组或者矩阵重复次数 如果第二个参数是数组 则数组内第一个数字为维度 第二个为重复次数
from numpy import tile, array, mat

target = [[4, 5, 6]]
dataSet = [[1, 2, 3], [2, 3, 4]]
sqDiffMat = tile(target, [2, 1]) - dataSet
diffMat = sqDiffMat ** 2
# axis=1 行中相加 axis=0 列中相加
distance = diffMat.sum(axis=1)
sqDistance = distance ** 0.5
print(sqDistance.argsort())
print(target[0].__len__())
classCount = {'index1': 3, 'index2': 9, 'index3': 1}
indexList = list(classCount.values())
index = mat(list(classCount.values())).argsort()
resultIndex = sorted(classCount.items(), key=lambda item: item[1])
print(resultIndex)
# print(resultIndex[0][0])
# print([v for v in classCount.values().argsort()])
