import xml.etree.ElementTree as et
import os
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from shutil import copy


import  cv2

import argparse
def Find_child_List(XML_path,nodeId,childList):
    tree =et.parse(XML_path)
    root = tree.getroot()

    for node_ in root.iter('node'):
        if 'no' in str(node_.attrib):
            if node_.find('id').text==nodeId :
                childList.append(node_.find('leftChild').text)
                childList.append(node_.find('rightChild').text)
                Find_child_List(XML_path,node_.find('leftChild').text, childList)
                Find_child_List(XML_path,node_.find('rightChild').text, childList)


    return childList

def Find_parent(XML_path,nodeId,parentID,times):
    tree = et.parse(XML_path)
    root = tree.getroot()

    for node_ in root.iter('node'):
        if 'no' in str(node_.attrib):
            if node_.find('leftChild').text==nodeId or node_.find('rightChild').text ==nodeId :
                parentID.append(node_.find('id').text)
                if times>0:
                    times-=1
                    Find_parent(XML_path,node_.find('id').text,parentID,times)
    return parentID


def Get_leaf_list(XML_path):
    tree =et.parse(XML_path)
    root = tree.getroot()
    idList=[]
    pathList=[]
    for node_ in root.iter('node'):
        if 'yes' in str(node_.attrib):
            idList.append(node_.find('id').text)
            pathList.append(node_.find('path').text)
    return idList,pathList



def Find_id(XML_path,path_name):
    tree = et.parse(XML_path)
    root = tree.getroot()
    ID=[]
    for node_ in root.iter('node'):
        if 'yes' in str(node_.attrib):
            if node_.find('path').text==path_name:
                ID.append(node_.find('id').text)
    return ID


def sampling(data,label,N_number):
    dataList = np.load(file=data)
    LabelList = np.load(file=label)
    mylist = list(range(len(LabelList)))
    #x = np.random.random_integers(0,10)
    x=random.sample(mylist,N_number)
    Datalist_R=[]
    LabelList_R=[]

    for each in x:
        Datalist_R.append(dataList[each])
        LabelList_R.append(LabelList[each])

    return Datalist_R,LabelList_R



dataList = np.load(file='Hog_data_binary.npy')
FullLabel=np.load(file="Hog_label_binary.npy")
idList,pathList=Get_leaf_list("images_Tree.xml")


def Ranking_similarity(query,List,number):

    sampling = random.sample(List, number)

    target = []
    distanceList = []
    for each in sampling:

        target.append([dataList[int(each)], each])
        for each1 in target:
            data = each1[0]
            similarity = cosine_similarity([query], [data])
            distanceList.append([FullLabel[int(each1[1])], similarity[0]])
    feature_dict = sorted(dict(distanceList).items(), key=lambda x: x[1], reverse=True)
    return feature_dict



#query=query Images path
#Tree_file=path
#datalist='Hog_data_binary.npy'
#
def searchImage(query,Tree_file,degrees):

    im = cv2.imread(query)
    halfImg = cv2.resize(im, (128, 128))
    hog = cv2.HOGDescriptor()
    query=hog.compute(halfImg).flatten()

    #sampling=idList[::100]


    findList=[]
    finalRanking=[]
    copylist=[]
    step=1
    sample=1
    if len(idList)>=100:
        step=10
        sample=5

    for target in  [idList[x:x+step] for x in range(0, len(idList), step)]:


        for i in Ranking_similarity(query,target,sample)[:1]:
            finalRanking.append(i)

            findList.append(i[0])

    finalRanking = sorted(dict(finalRanking).items(), key=lambda x: x[1], reverse=True)
    parent = []
    ParentList = []
    output = []
    parent.append(Find_parent(Tree_file, Find_id(Tree_file, str(finalRanking[:1][0][0]))[0], ParentList, degrees))

    for eachParent in ParentList:
        singlechild = []
        childlist = set(Find_child_List(Tree_file, eachParent, singlechild)).intersection(set(idList))
        for eachChild in childlist:
            copylist.append(FullLabel[int(eachChild)])
            output.append([FullLabel[int(eachChild)], cosine_similarity([query], [dataList[int(eachChild)]])])
            # print(FullLabel[int(eachChild)])

    for copyPic in finalRanking[:1]:
        copy('binary/' + copyPic[0],
             'testResult/' + copyPic[0])

    for copyPic1 in copylist:
        copy('binary/' + copyPic1,
             'testResult/' + copyPic1)

    output = sorted(dict(output).items(), key=lambda x: x[1], reverse=True)
    print(output)
    return output

parser = argparse.ArgumentParser(description='Find more Candidates by change the number of degree')
parser.add_argument('--QueryImage_dir', dest='Image_path', type=str, default='test.jpg', help='ex test.jpg')
parser.add_argument('--degree', dest='degree', type=int, default='0', help='0~')
args = parser.parse_args()
print(args.Image_path)

searchImage(args.Image_path,'images_Tree.xml',args.degree)