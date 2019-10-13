#coding=utf-8
import os

import  cv2

import argparse

def getfileFromfilter(rootdir):
    list = os.listdir(rootdir)
    ReturnList=[]
    for i in range(0, len(list)):
        if list[i]!='.DS_Store':

            path = os.path.join(rootdir, list[i])
            ReturnList.append(path)
    return (ReturnList)

def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary

def binarition(ImageDir):
    if os.path.isdir('binary'):
        print('exists')
    else:

        os.mkdir('binary')
    for object in getfileFromfilter(ImageDir):
        img = cv2.imread(object)
        newname = object.split('/')[1]
        cv2.imwrite('binary/'+str(newname), threshold_demo(img))

parser = argparse.ArgumentParser(description='Batch binarization of target folders')
parser.add_argument('--ImageData_dir', dest='Image_path', type=str, default='ImageData', help='ex ImageData')
args = parser.parse_args()
print(args.Image_path)
binarition(args.Image_path)