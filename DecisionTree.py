
# coding: utf-8

# In[30]:


import math
import copy
import uuid
import pickle
from collections import defaultdict, namedtuple


# ### 信息量
# 
# I = -log(P(x))
# 
# ### 信息熵
# 
# 信息量的期望值
# 
# H = -ΣP(x)log(P(x))

# In[20]:


def get_shanno_entorpy(values):
    '''
    根据给定列表中的值计算其Shanno Entropy
    '''
    uniq_vals = set(values)
    vals_num = {key: values.count(key) for key in uniq_vals}
    probs = [v/len(values) for k, v in vals_num.items()]
    entorpy = sum([-prob*math.log2(prob) for prob in probs])
    return entorpy


# In[22]:


values = [1,2,3,4,5,1,2,1,3,4,2,3,1,5,3]
get_shanno_entorpy(values)


# ### 信息增益
# 将数据集划分，划分后的数据集的信息熵是每个子集的信息熵的平均期望值
# HS = ΣP(D)H(D) = Σlen(D)/len(S)\* H(D)
# 
# 信息增益为 Gain = H - HS
# 
# ID3算法使用的就是基于信息增益的选择属性方法

# In[ ]:


def choose_best_split_feature1(dataset, classes):
    ''' 根据信息增益确定最好的划分数据的特征
    :param dataset: 待划分的数据集
    :param classes: 数据集对应的类型
    :return: 划分数据的增益最大的属性索引
    '''
    #注意：划分的数据集是classes（label）
    base_entorpy = get_shanno_entorpy(classes)
    feat_num = len(dataset[0])
    entorpy_gains = []
    for i in range(feat_num):
        splited_dict = split_dataset(dataset, classes, i)
        new_entorpy = sum([len(subclasses)/len(classes) * get_shanno_entorpy(subclasses) 
                           for _, (_, subclasses) in splited_dict.items()])
        entorpy_gains.append(base_entorpy - new_entorpy)
    return entorpy_gains.index(max(entorpy_gains))

#另一种无需计算H的实现
#因为H固定，所以最大化Gain，即最小化HS
def choose_best_split_feature2(dataset, classes):
    ''' 根据信息增益确定最好的划分数据的特征
    :param dataset: 待划分的数据集
    :param classes: 数据集对应的类型
    :return: 划分数据的增益最大的属性索引
    '''
    #注意：划分的数据集是classes（label）
    feat_num = len(dataset[0])
    entorpy_gains = []
    for i in range(feat_num):
        splited_dict = split_dataset(dataset, classes, i)
        new_entorpy = sum([len(subclasses)/len(classes) * get_shanno_entorpy(subclasses) 
                           for _, (_, subclasses) in splited_dict.items()])
        entorpy_gains.append(new_entorpy)
    return entorpy_gains.index(min(entorpy_gains))


# ### 增益比率
# 解决信息增益方法可能带来的无意义的分类，比如每个人有一个唯一的姓名，按姓名的feature分每类是最纯的，只有一个，这种分类无意义（无泛化能力，类似过拟合）。
# 方法：
# 引入一个分裂信息，分裂的越多，分裂信息会越大：SplitInfo = -Σlen(D)/len(S)log(len(D)/len(S))
# 
# 增益比率定义为：GainRadio = Gain/SplitInfo
# 
# 增益比率最大，则选择的属性最佳
# 
# **问题：分裂信息有可能为0，也有可能趋近于0，此时得到的增益比率无意义 **
# 
# 改进的措施就是在分母加一个平滑，这里加一个所有分裂信息的平均值：
# GainRadio = Gain/(SplitInfo_mean+SplitInfo)

# In[ ]:





# ### 基尼指数
# #### 基尼不纯度
# Gini(D) = 1 - ΣPi² （越纯值越小0）
# 
# #### 在CART(Classification and Regression Tree)算法中利用基尼指数构造二叉决策树
# 
# 属性R分裂后的基尼系数为：
# Gini_s(D, R) = Σlen(D)/len(S)Gini(D) (二分。多个分法的时候取Gini_s最小的分法)
# 
# 增量 ΔGini(R) = Gini(D) - Gini_s(D, R)
# 
# 取增量最大的属性作为最佳分裂属性
# 
# #### 若不构造二叉决策树
# 分裂后的不纯度为
# Gini_s(D) = Σlen(D)/len(S)Gini(D)
# 
# 增量计算同样。
# 

# In[ ]:





# ### 创建决策树
# 根据数据集及属性不断分裂（递归）
# 
# ** 终止条件：**
# 1、遍历完所有的属性
# 2、此数据集内只有一个类别的数据了
# 
# * 实现方法：用字典实现决策树的嵌套

# In[ ]:


def create_tree(dataset, classes, feat_names):
    ''' 根据当前数据集递归创建决策树
    :param dataset: 数据集
    :param feat_names: 数据集中数据相应的特征名称
    :param classes: 数据集中数据相应的类型
    :param tree: 以字典形式返回决策树
    '''
    #只有一种分类
    if len(set(classes)) == 1:
        return classes[0]
    #所有的feature都遍历完了,返回比例最多的类型
    if len(feat_names) == 0:
        return get_majority(classes)
    #分裂创建子树
    tree = {}
    best_feat_idx = choose_best_split_feature2(dataset, classes)
    feature = feat_names[best_feat_idx]
    tree[feature] = {}
    # 创建用于递归创建子树的子数据集
    sub_feat_names = feat_names[:]
    sub_feat_names.pop(best_feat_idx)
    split_dict = split_dataset(dataset, classes, best_feat_idx)
    for feat_val,(sub_dataset, sub_classes) in split_dict.items():
        tree[feature][feature_val] = create_tree(sub_dataset, sub_classes, sub_feat_names)
    
    #self.tree = tree
    #self.feat_names = feat_names
    
    return tree


# ### 构建决策树

# In[39]:


#完整的类
class DicisionTreeClassifier(object):
    
    def get_shanno_entorpy(self, values):
        '''
        根据给定列表中的值计算其Shanno Entropy
        '''
        uniq_vals = set(values)
        vals_num = {key: values.count(key) for key in uniq_vals}
        probs = [v/len(values) for k, v in vals_num.items()]
        entorpy = sum([-prob*math.log2(prob) for prob in probs])
        return entorpy
    
    def choose_best_split_feature2(self, dataset, classes):
        ''' 根据信息增益确定最好的划分数据的特征
        :param dataset: 待划分的数据集
        :param classes: 数据集对应的类型
        :return: 划分数据的增益最大的属性索引
        '''
        #注意：划分的数据集是classes（label）
        feat_num = len(dataset[0])
        entorpy_gains = []
        for i in range(feat_num):
            splited_dict = self.split_dataset(dataset, classes, i)
            new_entorpy = sum([len(subclasses)/len(classes) * self.get_shanno_entorpy(subclasses) 
                               for _, (_, subclasses) in splited_dict.items()])
            entorpy_gains.append(new_entorpy)
        return entorpy_gains.index(min(entorpy_gains))
    
    def create_tree(self, dataset, classes, feat_names):
        ''' 根据当前数据集递归创建决策树
        :param dataset: 数据集
        :param feat_names: 数据集中数据相应的特征名称
        :param classes: 数据集中数据相应的类型
        :param tree: 以字典形式返回决策树
        '''
        #只有一种分类
        if len(set(classes)) == 1:
            return classes[0]
        #所有的feature都遍历完了,返回比例最多的类型
        if len(feat_names) == 0:
            return get_majority(classes)
        #分裂创建子树
        tree = {}
        best_feat_idx = self.choose_best_split_feature2(dataset, classes)
        feature = feat_names[best_feat_idx]
        tree[feature] = {}
        # 创建用于递归创建子树的子数据集
        sub_feat_names = feat_names[:]
        sub_feat_names.pop(best_feat_idx)
        splited_dict = self.split_dataset(dataset, classes, best_feat_idx)
        for feat_val,(sub_dataset, sub_classes) in splited_dict.items():
            tree[feature][feat_val] = self.create_tree(sub_dataset, sub_classes, sub_feat_names)
    
        self.tree = tree
        self.feat_names = feat_names
    
        return tree
    
    @staticmethod
    def split_dataset(dataset, classes, feat_idx):
        """
        根据某个特征以及特征值划分数据集
        :param dataset: 待划分的数据集, 有数据向量组成的列表.
        :param classes: 数据集对应的类型, 与数据集有相同的长度
        :param feat_idx: 特征在特征向量中的索引
        :return splited_dict: 保存分割后数据的字典 特征值: [子数据集, 子类型列表]
        """
        splited_dict = {}
        for data_vect, cls in zip(dataset, classes):
            #对于不同的data_vect，feat_val的值是不同的，这样对应的data就会插入到不同的key下
            feat_val = data_vect[feat_idx]
            sub_dataset, sub_classes = splited_dict.setdefault(feat_val, [[], []])
            sub_dataset.append(data_vect[: feat_idx] + data_vect[feat_idx+1: ])  
            #去除data中已经分解过的属性的值
            sub_classes.append(cls)
        
        return splited_dict
    
    def get_majority(classes):
        """
        返回类中占据最多数的类型
        """
        cls_num = defaultdict(lambda:0)
        for cls in classes:
            cls_num[cls] += 1
        #注意用法：返回字典中最大的value对应的key
        return max(cls_num, key=cls_num.get)
    
    #分类函数
    def classify(self,test_vect, tree=None, feat_names=None):
        if tree is None:
            tree = self.tree
        if feat_names is None:
            feat_names = self.feat_names
        
        #结束条件
        if type(tree) is not dict:
            return tree
        
        feature = list(tree.keys())[0]
        value = test_vect[feat_names.index(feature)]
        subtree = tree[feature][value]
        return self.classify(test_vect, subtree, feat_names)
    
    #存储决策树
    def dump_tree(self, filename, tree=None):
        if tree is None:
            tree = self.tree
        with open(filename, 'wb') as f:
            pickle.dump(tree, f)
            
    #加载决策树
    def load_tree(self, filename):
        with open(filename, 'rb') as f:
            tree = pickle.load(f)
            self.tree = tree
        return tree
        


# In[40]:


#跑一个数据集看看

lense_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
X = []
Y = []

with open('lenses.txt', 'r') as f:
    for line in f:
        data = line.strip().split('\t')
        X.append(data[:-1])
        Y.append(data[-1])

clf = DicisionTreeClassifier()
clf.create_tree(X, Y, lense_labels)
clf.tree


# In[41]:


clf.classify(['pre','hyper','no','normal'])


# In[ ]:





# In[ ]:





# In[ ]:




