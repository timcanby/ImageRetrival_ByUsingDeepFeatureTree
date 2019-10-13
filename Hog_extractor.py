import os
import  cv2
import numpy as np

def getfileFromfilter(rootdir):
    list = os.listdir(rootdir)
    ReturnList=[]
    for i in range(0, len(list)):
        if list[i]!='.DS_Store':

            path = os.path.join(rootdir, list[i])
            ReturnList.append(path)
    return (ReturnList)


#=============Resize_Image===============
def resizeImage(path):
    img = np.array(cv2.imread(path))
    res = cv2.resize(img, (128, 128))

    return res

def pretreatment(ima):

    im = cv2.imread(ima)
    #im = cv2.imread('bunkatsu/' + str(eachPic))
    halfImg = cv2.resize(im, (128, 128))
    hog = cv2.HOGDescriptor()
    feature= hog.compute(halfImg)



    return feature.flatten()



data=[]
leaf_labels=[]
counter=0
#==================

for item in getfileFromfilter('binary'):
    data.append(pretreatment(item))
    leaf_labels.append(item.split('/')[1])
    counter+=1

np.save(file="Hog_data_binary.npy", arr=data)
np.save(file="Hog_label_binary.npy", arr=leaf_labels)