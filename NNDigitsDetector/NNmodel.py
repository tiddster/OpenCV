import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.model_selection as ms
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import  Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D

#####################################
path = 'dataset'
imageDimensions = (32, 32, 3)
#####################################

"""
读取所有图片
"""
images = []
classNames = []
digitsList = os.listdir(path)

print("正在读取图像......")
for folderName in tqdm(digitsList):
    picList = os.listdir(path + '/' + folderName)
    for picFileName in picList:
        img = cv2.imread(path + '/' + folderName + '/' + picFileName)
        img = cv2.resize(img, (32, 32))
        images.append(img)
        classNames.append(folderName)
print("图像读取完毕")

images = np.array(images)
classNames = np.array(classNames)

"""
清洗数据：拆分数据集，并检查训练集是否划分均匀，用条形图展示
"""


def split_dataset(images, classNames):
    pass


x_train, x_test, y_train, y_test = ms.train_test_split(images, classNames, test_size=0.2)
x_train, x_cross, y_train, y_cross = ms.train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape)

numsOfSample = []
for i in digitsList:
    numsOfSample.append(len(np.where(y_train == i)[0]))
print(f"训练集中0~9的比例为：{numsOfSample}")

plt.figure(figsize=(10, 5))
plt.bar(range(0, len(numsOfSample)), numsOfSample)
plt.ylabel("Number of Images")
plt.xlabel("Class Names")
plt.show()


"""
数据清晰：预处理，灰度化和正则化
"""


def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


# map(fun, xs) = for  x in xs: fun(x)
"""
数据清洗：先将数据转化为np中数组形式，再对其进行旋转、对称等增强处理
"""
x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_cross = np.array(list(map(preProcess, x_cross)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_cross = x_cross.reshape(x_cross.shape[0], x_cross.shape[1], x_cross.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(x_train)

y_train = to_categorical(y_train, len(numsOfSample))
y_test = to_categorical(y_test, len(numsOfSample))
y_cross = to_categorical(y_cross, len(numsOfSample))


"""
搭建网络
"""
def netModel():
    numOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    numOfNode = 500

    model = Sequential()
    model.add((Conv2D(numOfFilters, sizeOfFilter1,
                      input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(numOfFilters, sizeOfFilter1,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(numOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(numOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(numOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(numsOfSample), activation='relu'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = netModel()
print(model.summary())