'''
Name:
        构造一棵决策树
Author:
        Booksort
Time:
        2023-3-25
Function:
        由于决策树的构造是在训练过程中搭建的，所以这也是决策树训练模块
Think:
        递归构造决策树,对于根节点，先所有的决策变量L，依次找出变量中的特征个数，然后对于各个特征预分类的集合计算熵或基尼系数，再求加权平均值得到切分后的熵值
        再依次比较计算找到信息增益最大的决策变量作为根节点...以此类推对于根节点分割出的n个特征子集合再次对于剩下的L-1个决策变量进行计算找到最大的信息增益

        递归时，先获取要分类的数据集，在对于各个决策变量与分类的集合进行切分然后计算，然后将这一层的已经分好类的集合进行”保存“，再进入下一层递归

        对于最后构造的决策树结构，是以 字典嵌套字典的结构 进行维护的
        每完成一层节点的构建，数据的特征变量就要少一个维度，可以减少内存消耗

        对于递归构建决策树的流程思考：
            是否符合结束递归的条件：1、数据集合中的分类结果尤其只有一种，2、所有的特征变量都被用于分类决策了，数据集中只剩下分类结果
            不符合继续分割数据集，选取剩余决策变量中，信息增益最大的决策变量作为改决策点。作为决策点的子节点，如果不是连续值，几个子节点就是决策变量的那几个离散取值
          同时，要记录那几个子节点，递归回来时要进行分割，有点像树的前序遍历：根节点-左子树-右子树
'''
import numpy as np


global Bs_DecTree  # 字典树


def ImportDataset():

    Dataset = np.array((
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [2, 1, 0, 0, 1],
        [2, 2, 1, 0, 1],
        [2, 2, 1, 1, 0],
        [1, 2, 1, 1, 1],
        [0, 1, 2, 0, 0]
    ))
    Classname = {"outlook": 0, "temperature": 1, "humidity": 2, "windy": 3, "play": 4}

    # Class_if_used = [0, 0, 0, 0]        # 判断某个决策变量是否已经被使用于构造节点，在递归时用于判断

    return Dataset, Classname

def CountMostDivers(unionlist): # 统计该集合中哪一类别占比最多

    return np.argmax(np.bincount(unionlist))


def calcuEntropy(colVec):   # 对于集合中的分类结果的列向量
    divNum = set(list(colVec))  # 统计分类的类别数
    numexams = len(colVec)      # 统计集合中的样本数目
    entropy = 0 # 熵值
    for a in divNum:

        elem = len([val for val in colVec if val == a])  # 统计出现指定类别的数量做分子
        prob_a = elem/numexams
        entropy = entropy - prob_a * np.log2(prob_a)

    return entropy          # 返回计算的集合的熵值


## 功能完成
def bestDecisionVarOfInformGain(dataset, VarName):  # param：多列的矩阵，包括分类结果；决策变量字典树，名字与矩阵索引
    # 统计待分类的决策变量
    decVarli = list(VarName.keys())
    decVarli.pop()
    # 计算当前待分类的集合的熵值
    baseEntropy = calcuEntropy(dataset[:, -1])  # 待切分的集合的熵值
    # 依次遍历计算决策变量的特征分类
    recordEntr = {}
    for a in decVarli:
        # 提取矩阵的列
        Xarray = dataset[:, VarName[a]]
        featVal = set(list(Xarray))
        EntrSum = 0
        for feat in featVal:
            collect = np.array([dataset[index, -1] for index in range(len(Xarray)) if Xarray[index] == feat])
            Entr_i = calcuEntropy(collect)
            EntrSum = EntrSum + Entr_i * (collect.size / Xarray.size)

        recordEntr[a] = EntrSum
    Gain = dict([(key, baseEntropy-val) for key, val in recordEntr.items()])
    maxGain = max(Gain.values())
    for k, v in Gain.items():
        if v >= maxGain:
            bestFeatName = k
            break
    # 统计完所有的决策变量分类的信息熵，在统计最大的信息增益
    return bestFeatName     # 返回被用于分类的决策变量的下标


def subDataSplit(dataset, featIndex, var):  # 参数:带分割的数据集，依据决策变量进行分割，决策变量中的种类,列不用删除，但是行要分割
    retDataSet = []
    for index in range(dataset[:, featIndex].size):
        if dataset[index, featIndex] == var:
            retDataSet.append(dataset[index, :])

    return np.array(retDataSet)

def ConstructTree_Recursion(dataset, Varname, featurelist):
    datalabels = [lab[-1] for lab in dataset]   # 获取取数据中的标签值，也就是决策变量导致的分类结果
    # print(datalabels)
    if len(set(datalabels)) == 1:  # set可以制作一个集合（保证集合中没有相同元素）如果集合元素数量只有1个说明data的标签都是相同的，熵值是0，可以结束递归
        print("该集合熵值为0,分类结果为:", datalabels[0])
        return datalabels[0]
    if len(dataset[0]) == 1:   # 每建立一个节点就要删除一个特征变量，即数据集的维度变成n*1,就剩下标签维度了，那么就结束分裂
        return CountMostDivers(dataset[:, -1])#对于剩下的数据集，没有特征变量了，就对于数据集中的分类结果取众数作为改叶子节点的分类结果

    featName = bestDecisionVarOfInformGain(dataset, Varname) # 将数据集选取最大的决策变量影响的信息增益，将数据集用决策变量的取值进行切分 返回选取的特征变量的字典名
    print("分类的决策变量为", featName)
    decTree = {Varname[featName]: {}}

    featIndex = Varname[featName]
    # 贪心算法，先算到局部最大的，就直接取走，对于子节点的兄弟节点肯恶搞并不是全局最优的
      # 删除那一列特征变量的索引，减少消耗,
    # 遍历这一层子节点的兄弟节点,即特征变量有几个取值，代表可以分成几类
    valdiv = set(dataset[:, Varname[featName]]) # 获取这特征变量中断取值数量
    del Varname[featName]

    for var in valdiv:
        subdata = subDataSplit(dataset, featIndex, var)
        print("分类特征：", var)
        print("分割后的数据集\n", subdata)

    # for var in valdiv:
    #     subdata = subDataSplit(dataset, featIndex, var)
    #
    #     decTree[featIndex][var] = ConstructTree_Recursion(subdata, Varname, featurelist)

    return decTree




if __name__ == "__main__":
    [dataset, Classname] = ImportDataset()
    result = ConstructTree_Recursion(dataset, Classname,[])

