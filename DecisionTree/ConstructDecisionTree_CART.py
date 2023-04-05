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


def proGini(ulist):   # 计算基尼系数最优值
    diverselist = [dataset[index, -1] for index in ulist]
    # 对于离散值都当作连续纸一起处理，对于集合中的分类结果先排序，进行相邻两个元素取中值，作为划分点
    diverselist.sort()
    segmenValList = set([(diverselist[index]+diverselist[index+1])/2 for index in range(len(diverselist)-1)])
    bestprob = []
    for segmenVal in segmenValList:
        prob1 = len([val for val in diverselist if val <= segmenVal]) / len(diverselist)
        prob2 = len([val for val in diverselist if val > segmenVal]) / len(diverselist)
        bestprob.append(prob1**2 + prob2**2)

    return max(bestprob)    # 返回分界点最大的值，则计算得到的基尼系数就会最小，就说明是这个决策变量最好的分类效果




def bestDecisionVarOfInformGain(dataset, VarName):   # 返回值，计算后信息增益最大的那个决策变量
    # 依次选这决策变量的
    li = list(VarName.keys())
    li.pop()
    Gi = []
    for a in li:
        # 提取矩阵的列
        Xarray = dataset[:, VarName[a]]
        # 计算该决策变量中的最大的基尼系数---------------CART算法，有问题，逻辑还得改动
        Gi.append(1 - proGini(Xarray))
        '''
        for b in set(Xarray):
            # 对于各特征变量的取值进行分类，分别统计集合中的分类结果
            bl = [ind for ind, val in enumerate(list(Xarray)) if val == b]
            # 还要统计集合中其分类结果的概率
            gini = 1 - proGini(bl)
            Gi.append(gini)
        '''
    return Gi.index(max(Gi))    # 返回被用于分类的决策变量的下标

def subDataSplit(dataset, featIndex, var):  # 参数:带分割的数据集，依据决策变量进行分割，决策变量中的种类

    return

def ConstructTree_Recursion(dataset, Varname, featurelist):
    datalabels = [lab[-1] for lab in dataset]   # 获取取数据中的标签值，也就是决策变量导致的分类结果
    print(datalabels)
    if len(set(datalabels)) == 1:  # set可以制作一个集合（保证集合中没有相同元素）如果集合元素数量只有1个说明data的标签都是相同的，熵值是0，可以结束递归
        print("该集合熵值为0:", datalabels[0])
        return datalabels[0]
    if len(dataset[0]) == 1:   # 每建立一个节点就要删除一个特征变量，即数据集的维度变成n*1,就剩下标签维度了，那么就结束分裂
        return CountMostDivers(dataset[:, -1])#对于剩下的数据集，没有特征变量了，就对于数据集中的分类结果取众数作为改叶子节点的分类结果

    featIndex = bestDecisionVarOfInformGain(dataset, Varname) # 将数据集选取最大的决策变量影响的信息增益，将数据集用决策变量的取值进行切分 返回选取的特征变量的下标
    decTree = {Varname[featIndex]: {}}
    # 贪心算法，先算到局部最大的，就直接取走，对于子节点的兄弟节点肯恶搞并不是全局最优的
    del classname[featIndex]  # 删除那一列特征变量的索引，减少消耗,
    # 遍历这一层子节点的兄弟节点、
    # valdiv = set(dataset[:][featIndex]) # 获取这特征变量中断取值数量
    for var in range(2):
        decTree[Varname[featIndex]][var] = ConstructTree_Recursion(subDataSplit(dataset, featIndex, var), Varname, featurelist)

    return decTree




if __name__ == "__main__":
    [dataset, classname] = ImportDataset()
    print(type([lab[-1] for lab in dataset]))
    print(np.argmax(np.bincount(dataset[:, -1])))

    ConstructTree_Recursion(dataset, classname,[])
