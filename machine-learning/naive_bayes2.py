# -- coding: utf-8 --
import numpy as np
from math import log


def loadDataSet():
    # 创建单词向量及对应的分类，1代表侮辱性文字，0代表正常言论
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):  # 创建一个过滤dataSet重复数据的表
    vocabSet = set()  # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet.union(set(document))  # 创建两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):  # 将文档转换成特征向量 (词集模型)
    returnVec = [0] * len(vocabList)  # 创建一个长度与不重复词表一样的一维数组，元素默认为0
    for word in inputSet:
        try:
            # 查找单词在词表中的索引
            word_index = vocabList.index(word)
            returnVec[word_index] = 1  # 若词表单词在文档中出现过，则将元素改为1
        except ValueError:
            # 单词不在词表中时打印，使用 Python 3 的 print 函数
            print(f"the word: {word} is not in my Vocabulary!")
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 计算训练样本数量
    numWords = len(trainMatrix[0])  # 计算不重复词表中单词数量

    # 类别为1的训练样本的概率，即P(Y=c1)
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)
    print("pAbusive = " + str(pAbusive))

    # 初始化计数器，使用拉普拉斯平滑 (Laplace Smoothing)
    # 所有次数初始化为1，所有分母初始化为2 (拉普拉斯平滑的参数 $\lambda=1$)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 类别1的词频向量
            p1Denom += np.sum(trainMatrix[i])  # 类别1的总词数
        else:
            p0Num += trainMatrix[i]  # 类别0的词频向量
            p0Denom += np.sum(trainMatrix[i])  # 类别0的总词数
    print("p0Num = " + str(p0Num))
    print("p1Num = " + str(p1Num))
    print("p0Denom = " + str(p0Denom))
    print("p1Denom = " + str(p1Denom))

    # 计算对数条件概率：log(P(x=xi|Y=ck))
    # 乘法转加法，防止下溢，提高数值稳定性
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    print("p0Vect = " + str(p0Vect))
    print("p1Vect = " + str(p1Vect))

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):

    print("vec2Classify = " + str(vec2Classify))

    # p1 = log(P(x1|Y=c1)) + ... + log(P(xn|Y=c1)) + log(P(Y=c1))
    # vec2Classify * p1Vec 利用了NumPy的按元素乘法，只保留了测试样本中出现的单词的对数概率
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)

    # p0 = log(P(x1|Y=c0)) + ... + log(P(xn|Y=c0)) + log(P(Y=c0))
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)

    # 最大后验概率 (MAP) 决策：哪个概率大，就属于哪一类
    if p1 > p0:
        return 1  # 侮辱性言论
    else:
        return 0  # 正常言论


def testingNB():
    listOPosts, listClasses = loadDataSet()  # 获取单词向量及对应分类

    myVocabList = createVocabList(listOPosts)  # 获取不重复的词表
    print("myVocabList = " + str(myVocabList))
    print("size = " + str(len(myVocabList)))

    trainMat = []
    for postinDoc in listOPosts:
        # 输入某文档，输出文档向量，向量为1或0 (词集模型)
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print("trainMat = " + str(trainMat))
    print("size = " + str(trainMat.__len__()) + " * " + str(trainMat[0].__len__()))
    # 类似 [[0,0,0,1,0,1][1,0,1,1,0,0]..]，在单词表中存在即是1，不存在就是0

    # 训练模型：计算对数条件概率和先验概率
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

    # --- 测试案例 1 ---
    testEntry1 = ['love', 'my', 'dalmation']
    thisDoc1 = np.array(setOfWords2Vec(myVocabList, testEntry1))

    # 使用 Python 3 的 print 函数
    print(f"测试文本: {testEntry1}")
    print(f"预测分类: {classifyNB(thisDoc1, p0V, p1V, pAb)} (0:正常, 1:侮辱性)")
    print("-" * 30)

    # --- 测试案例 2 ---
    testEntry2 = ['stupid', 'garbage']
    thisDoc2 = np.array(setOfWords2Vec(myVocabList, testEntry2))

    # 使用 Python 3 的 print 函数
    print(f"测试文本: {testEntry2}")
    print(f"预测分类: {classifyNB(thisDoc2, p0V, p1V, pAb)} (0:正常, 1:侮辱性)")


# 执行测试
if __name__ == '__main__':
    testingNB()