import cv2
import numpy as np
from MyOpenCVLib import myLib

#####################################
capture = cv2.VideoCapture(0)
#####################################

while True:
    success, img = capture.read()

    resContours, imgContour = myLib.getContours(img, filterPoints=4, minArea=1000, contourType=myLib.CONTOURTS_POINTS_LINE)

    if len(resContours) != 0:
        biggestContour = resContours[0][2]
        myLib.drawRect(imgContour, biggestContour, 3)

    key = cv2.waitKey(1)
    cv2.imshow("xxx", myLib.stackImages([img, imgContour]))
    if key == 27:
        break