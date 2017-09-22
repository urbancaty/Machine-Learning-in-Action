from math import log
import operator

def calEnt(dataSet):    # caculate entropy
    ent = 0
    dir = {}
    for i in range(len(dataSet)):
        if dataSet[i][-1] not in dir:
            dir[dataSet[i][-1]] = 0
        dir[dataSet[i][-1]] += 1
    for key in dir:
        ent -= dir[key]/len(dataSet)*log(dir[key]/len(dataSet),2)
    return ent

def split(dataSet, axis, val):    # split dataSet according to value of feature in the specific axis
    newdataSet = []
    for vec in dataSet:
        if vec[axis] == val:
            temp = vec[:axis]
            temp.extend(vec[axis+1:])    # how to use extend()
            newdataSet.append(temp)
    return newdataSet

def bestFea(dataSet):    # choose the best feature according to information gain
    nFea = len(dataSet[0])-1
    ent = calEnt(dataSet)
    bestGain = 0
    bestFea = -1
    for i in range(nFea):
        vals = []
        for j in range(len(dataSet)):
            if dataSet[j][i] not in vals:
                vals.append(dataSet[j][i])
        for val in vals:
            feaEnt = calEnt(split(dataSet, i, val))
            if ent - feaEnt > bestGain :
                bestGain = ent - feaEnt
                bestFea = i
    return bestFea

def major(labels):    # figure out the label which has the hightest frequency
    count = {}
    for label in labels:
        if label not in count:
            count[label] = 0
        count[label] += 1
    sortedCount = sorted(count.items, key=operator.itemgetter(1), reverse=True)
    return sortedCount[0][0]

def createTree(dataSet, labels):
    list = [data[-1] for data in dataSet]
    if list.count(list[0]) == len(list):
        return list[0]
    if len(dataSet[0]) == 1:
        return major(list)
    feaIdx = bestFea(dataSet)
    label = labels[feaIdx]
    tree = {label : {}}
    del labels[feaIdx]
    feaVal = set([data[feaIdx] for data in dataSet])
    for val in feaVal:
        subLabels = labels[:]
        tree[label][val] = createTree(split(dataSet, feaIdx, val), subLabels)
    return tree

def classify(tree, labels, testVec):
    label = list(tree.keys())[0]
    tree = tree[label]
    feaIdx = labels.index(label)    # have to get the index of feature first
    for key in list(tree.keys()):
        if key == testVec[feaIdx]:
            if isinstance(tree[key],dict):    #judge if tree[key] is a 'dict' variance
                return classify(tree[key], labels, testVec)
            else:
                return tree[key]

def storeTree(tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'r')
    return pickle.load(fr)

'''
dataSet = [[1,1,'yes'], [1,1,'yes'], [1,0,'no'],[0,1,'no'],[0,1,'no']]
labels = ['no surfacing', 'flippers']
tree = createTree(dataSet, labels)
labels = ['no surfacing', 'flippers']
res = classify(tree, labels, [1,1])
'''

fr = open('lenses.txt')
dataSet = [line.strip().split('\t') for line in fr.readlines()]
labels = ['age', 'prescript', 'astigmatic', 'tearRate']
tree = createTree(dataSet, labels)
print(tree)
exit = 0
