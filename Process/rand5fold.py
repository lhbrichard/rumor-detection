import random
from random import shuffle
import os

cwd=os.getcwd()
# set 5 fold therefore the ratio of test is 0.2
# such method can guarantee that the number of each cate is same
def each_cate(li):
    length = int(len(li)*0.2)
    # each tuple stands (train,test)
    fold1,fold2,fold3,fold4,fold5 = (li[length:],li[:length]),\
                                    (li[:length]+li[length*2:],li[length:length*2]),\
                                    (li[:length2]+li[length*3:],li[length*2:length*3]),\
                                    (li[:length3]+li[length*4:],li[length*3:length*4]),\
                                    (li[:length4],li[length*4:length*5])
    return fold1,fold2,fold3,fold4,fold5
def fold_data_get(all_li):
    fold1_test,fold2_test,fold3_test,fold4_test,fold5_test = [],[],[],[],[]
    fold1_train,fold2_train,fold3_train,fold4_train,fold5_train = [],[],[],[],[]
    for li in all_li:
        fold1,fold2,fold3,fold4,fold5 = each_cate(li)
        # 1
        fold1_train.extend(fold1[0])
        fold1_test.extend(fold1[1])
        # 2
        fold2_train.extend(fold2[0])
        fold2_test.extend(fold2[1])
        # 3
        fold3_train.extend(fold3[0])
        fold3_test.extend(fold3[1])
        # 4
        fold4_train.extend(fold4[0])
        fold4_test.extend(fold4[1])
        # 5
        fold5_train.extend(fold5[0])
        fold5_test.extend(fold5[1])
    return fold1_test,fold1_train,\
            fold2_test,fold2_train,\
            fold3_test,fold3_train,\
            fold4_test,fold4_train,\
            fold5_test,fold5_train
            
def load5foldData(obj):
    if 'Twitter' in obj:
        labelPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_label_All.txt")
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        print("loading tree label" )
        NR,F,T,U = [],[],[],[]
        l1=l2=l3=l4=0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            label, eid = line.split('\t')[0], line.split('\t')[2]
            labelDic[eid] = label.lower()
            if label in labelset_nonR:
                NR.append(eid)
                l1 += 1
            if labelDic[eid] in labelset_f:
                F.append(eid)
                l2 += 1
            if labelDic[eid] in labelset_t:
                T.append(eid)
                l3 += 1
            if labelDic[eid] in labelset_u:
                U.append(eid)
                l4 += 1
        print(len(labelDic))
        print(l1,l2,l3,l4)
        # each cate inside needs shuffle
        random.seed(123)
        random.shuffle(NR)
        random.shuffle(F)
        random.shuffle(T)
        random.shuffle(U)
        # get 5 fold data
        fold1_test,fold1_train,\
            fold2_test,fold2_train,\
            fold3_test,fold3_train,\
            fold4_test,fold4_train,\
            fold5_test,fold5_train = fold_data_get((NR,F,T,U))
        
    if obj == "Weibo":
        labelPath = os.path.join(cwd,"data/Weibo/weibo_id_label.txt")
        print("loading weibo label:")
        F, T = [], []
        l1 = l2 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid,label = line.split(' ')[0], line.split(' ')[1]
            labelDic[eid] = int(label)
            if labelDic[eid]==0:
                F.append(eid)
                l1 += 1
            if labelDic[eid]==1:
                T.append(eid)
                l2 += 1
        print(len(labelDic))
        print(l1, l2)
        random.seed(123)
        random.shuffle(F)
        random.shuffle(T)

        # get 5 fold data
        fold1_test,fold1_train,\
            fold2_test,fold2_train,\
            fold3_test,fold3_train,\
            fold4_test,fold4_train,\
            fold5_test,fold5_train = fold_data_get((F,T))
        
    # each fold inside needs shuffle
    shuffle(fold1_test)
    shuffle(fold1_train)
    shuffle(fold2_test)
    shuffle(fold2_train)
    shuffle(fold3_test)
    shuffle(fold3_train)
    shuffle(fold4_test)
    shuffle(fold4_train)
    shuffle(fold5_test)
    shuffle(fold5_train)

    return fold1_test,fold1_train,\
            fold2_test,fold2_train,\
            fold3_test,fold3_train,\
            fold4_test,fold4_train,\
            fold5_test,fold5_train


from Process.process import loadTree    
import numpy as np
def loadbaselineData(obj):
    treeDic=loadTree(obj)
    x = np.array([]).reshape(0,5000)
    y = []
    if 'Twitter' in obj:
        labelPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_label_All.txt")
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        print("loading tree label" )
        NR,F,T,U = [],[],[],[]
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            label, eid = line.split('\t')[0], line.split('\t')[2]
            labelDic[eid] = label.lower()
            if len(treeDic[eid]) < 2 or len(treeDic[eid]) > 100000:
                continue
            data=np.load('data/'+obj+'graph/'+eid+".npz", allow_pickle=True)
            x = np.vstack((x,np.mean(data['x'],axis=0).reshape(1,-1)))
            if label in labelset_nonR:
                y += [0]
            if labelDic[eid] in labelset_f:
                y += [1]
            if labelDic[eid] in labelset_t:
                y += [2]
            if labelDic[eid] in labelset_u:
                y += [3]
        return x,np.array(y)
