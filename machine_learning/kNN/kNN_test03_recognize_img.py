from numpy import *
from os import listdir

from machine_learning.kNN.kNN_test01 import classify0


def img2vector(img):
    returnVector = zeros((1, 32 * 32))
    with open(img) as fr:
        for i in range(32):
            imgData = fr.readline()
            for j in range(32):
                returnVector[0, 32 * i + j] = int(imgData[j])
        return returnVector


def file2matrix(listname, testlistname):
    """
    训练集
    """
    imgList = listdir(listname)
    labels = []
    dataSetSize = len(imgList)
    trainingMat = zeros((dataSetSize, 32 * 32))
    for i in range(dataSetSize):
        imgName = imgList[i]
        labels.append(imgName.split("_")[0])
        imgPath = listname + "/" + imgName
        vector = img2vector(imgPath)
        trainingMat[i, :] = vector
    """
    测试集
    """
    testImgList = listdir(testlistname)
    testLabels = []
    testDataSetSize = len(testImgList)
    testMat = zeros((testDataSetSize, 32 * 32))
    for i in range(testDataSetSize):
        imgName = testImgList[i]
        testLabels.append(imgName.split("_")[0])
        testMat[i, :] = img2vector(testlistname + "/" + imgName)
    error = 0
    for i in range(testDataSetSize):
        result = classify0(testMat[i, :], trainingMat[:], labels[:], 5)
        if testLabels[i] != result: error += 1
        print("the kNN came back is:%f, the real answer is:%f" % (float(result), float(testLabels[i])))
    print("the error count is:%f" % (error))
    print("the error rate is:%f" % (error / float(testDataSetSize)))


if __name__ == '__main__':
    print(file2matrix("./trainingDigits", "./testDigits"))
