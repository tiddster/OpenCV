import cv2
import numpy as np
import os
import FeatureDetect
import myLib

#####################################
featureNum = 200
pukeW, pukeH = 285, 435
path = 'pukeImage1'

imageTest = cv2.imread('king_diamond_test.jpg')
imageTest = cv2.resize(imageTest, (pukeW, pukeH))
imageTest = cv2.GaussianBlur(imageTest, (5,5), 10)

capture = cv2.VideoCapture(0)

images, names = myLib.readImagesAndNamesFromLib(path, [])
#####################################
print("开始寻找特征")
orb = FeatureDetect.getORB(featureNum)
keyPtsList, desList = FeatureDetect.train_KeyPointsAndDescriptor(orb, images)
print("寻找特征完成")

while True:
    success, imageCap = capture.read()

    keyPtsCap, desCap = orb.detectAndCompute(imageCap, None)
    res = f"Fail, {len(keyPtsCap)}"

    if desCap is not None and len(desCap) >= featureNum // 2:
        bestIndex, bestMatches = FeatureDetect.findBestMatches(desCap, desList, 10)
        if bestIndex == -1:
            res = f"Match Fail, {len(bestMatches)}"
        else:
            res = names[bestIndex]

    cv2.putText(imageCap, res, (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    cv2.imshow("x", imageCap)
    key = cv2.waitKey(1)
    if key == 27:
        break
# for index, image in enumerate(images):
#     image = cv2.drawKeypoints(image, keyPtsList[index], None)
#     cv2.imshow("x", image)
#     cv2.waitKey(0)
