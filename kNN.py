from numpy import *
import operator

def createData():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# 计算数据间的欧氏距离，选取最近的k个，统计类别，输出统计频率最高的
def classify(point, data, labels, k):
    dataSize = data.shape[0]
    diff = data - tile(point, (dataSize,1))
    dist = sum(diff**2, axis = 1)**0.5
    index = argsort(dist)
    count = {}
    for i in range(k):
        label = labels[index[i]]
        count[label] = count.get(label, 0)+1
    sortedCount = sorted(count.items(), key = operator.itemgetter(1), reverse = True)
    return sortedCount[0][0]

def file2matx(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    nLines = len(arrayLines)
    matx = zeros((nLines, 3))
    labels = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        list = line.split('\t')
        matx[index,:] = list[0:3]
        labels.append(int(list[-1]))
        index += 1
    return matx, labels

# 归一化数据
def norm(data):
    min = data.min(0)
    max = data.max(0)
    interval = max - min
    size = data.shape[0]
    normedData = (data - tile(min, (size, 1)))/tile(interval, (size, 1))
    return normedData, min, interval

# 这里可以挑选不同的k计算错误率
def test():
    ratio = 0.2
    matx, labels = file2matx('datingTestSet2.txt')
    normMatx, min, interval = norm(matx)
    size = normMatx.shape[0]
    nTest = int(size*ratio)
    nError = 0
    for i in range(nTest):
        result = classify(normMatx[i], normMatx, labels, 4 )
        if(result != labels[i]): nError += 1
    print(nError/nTest)

# 给定一个人的特征，预测喜欢的程度
def person():
    result = ['not at ll', 'maybe', 'likely']
    game = float(input('percentage of time spent on video games'))
    miles = float(input('frequent flier miles earned per year'))
    iceCream = float(input('liters of ice cream consumed per year'))
    inArray = array([game, miles, iceCream])
    matx, labels = file2matx('datingTestSet2.txt')
    normMatx, min, interval = norm(matx)
    resIdx = classify((inArray - min)/interval, normMatx, labels, 4)
    print(result[resIdx-1])

#将手写数字转换成向量
def file2vec(filename):
    fr = open(filename)
    vec = zeros((1,1024))
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vec[0,i*32+j] = int(line[j])
    return vec

#手写数字测试
def hwtest():
    hwLabels = []
    trainList = listdir('trainingDigits')
    m = len(trainList)
    trainMatx = zeros((m,1024))
    for i in range(m):
        fileName = trainList[i]
        classNum = int(fileName.split('_')[0])
        hwLabels.append(classNum)
        trainMatx[i,:] = file2vec('trainingDigits\\'+fileName)
    testList = listdir('testDigits')
    nError = 0
    mTest = len(testList)
    for j in range(mTest):
        fileName = testList[j]
        classNum = int(fileName.split('_')[0])
        if(classNum != classify(file2vec('testDigits\\'+fileName), trainMatx, hwLabels, 4)):
            nError += 1
    print('the total error rate is %f' % (nError/mTest))


#hwtest()
#test()
#person()
