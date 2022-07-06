import numpy as np
import cv2
import os
from tqdm import tqdm
import sklearn

#####################################
path = 'dataset'
#####################################

"""
读取所有图片
"""
images = []
classNames = []
digitsList = os.listdir(path)

print("正在读取图像......")
for folderName in tqdm(digitsList):
    picList = os.listdir(path+'/'+folderName)
    for picFileName in picList:
        img = cv2.imread(path+'/'+folderName+'/'+picFileName)
        img = cv2.resize(img, (32, 32))
        images.append(img)
        classNames.append(folderName)
print("图像读取完毕")

images = np.array(images)
classNames = np.array(classNames)


"""
清洗数据：拆分数据集
"""

