import os
import  cv2
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
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
    res = cv2.resize(img, (224, 224))

    return res


base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
#print(model.summary())
def pretreatment(ima):

    img_path = ima
    print(ima)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    block4_pool_features = model.predict(x)

    return block4_pool_features.flatten()


data=[]
leaf_labels=[]
counter=0
#==================

for item in getfileFromfilter('binary'):
    data.append(pretreatment(item))
    leaf_labels.append(item.split('/')[1])
    counter+=1
np.save(file="data_binary.npy", arr=data)
np.save(file="label_binary.npy", arr=leaf_labels)