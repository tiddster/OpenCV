import cv2
import numpy as np
import dlib
import myLib
import FacialCharacterDetector

#####################################
img = cv2.imread('111.png')
scale = 1
wImg, hImg = 360, 360
#####################################

img = cv2.resize(img, (wImg, hImg))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgMark = FacialCharacterDetector.getFacialDetector(imgGray, img)

imgArray = [
    [img, imgMark]
]

cv2.imshow("img", myLib.stackImages(imgArray,0.8))
cv2.waitKey(0)