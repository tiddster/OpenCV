import cv2
import numpy as np
from FRnet import *
import os
from tqdm import tqdm
import face_recognition
import time


path = 'Images'
images = []
classNames = []
playerList = os.listdir(path)
print(playerList)

for fileName in playerList:
    curImg = cv2.imread(f"{path}\\{fileName}")
    images.append(curImg)
    classNames.append(fileName.split('.')[0])
    cv2.imshow(fileName.split('.')[0], curImg)
    cv2.waitKey(0)

print("读取完成")

encodes = []
for img in tqdm(images):
    encode = faceLocAndEncode(img)
    encodes.append(encode)

print("训练完成")

capture = cv2.VideoCapture(0)
while True:
    success, img = capture.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)

    cv2.imshow("xx", img)
    key = cv2.waitKey(1)

    encode = faceLocAndEncode(img)

    if encode is not None:
        res = compare_faces(encodes, encode)

        import time

        now = time.localtime()
        nowt = time.strftime("%Y-%m-%d-%H:%M:%S", now)  # 这一步就是对时间进行格式化

        index = -1
        for i, r in enumerate(res):
            if r:
                index = i
                break
        if index == -1:
            print(f"陌生人 {nowt}")
            cv2.putText(img, "陌生人", (0,0), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255))
        else:
            print(classNames[index] + f"  {nowt}",)
            cv2.putText(img, classNames[index] + f"  {nowt}", (0, 0), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))

        if key == 27:
            break

