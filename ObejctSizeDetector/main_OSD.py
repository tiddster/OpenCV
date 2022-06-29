import cv2
import numpy as np
import myLib
"""
在A4纸上放卡片，测量卡片的长款
A4纸作为背景
首先识别背景， 再在背景中识别卡片
"""
#####################################
scale = 1.6
pad = 20
capture = cv2.VideoCapture(0)
wImg, hImg = int(210 * scale), int(297 * scale)
#####################################

while True:
    #img = cv2.imread('222.png')
    success, img = capture.read()
    img = cv2.resize(img, (wImg, hImg))

    # todo: 识别背景
    # resContours:包含img中多个识别出来的边缘元组
    resContours, imgContour = myLib.getContours(img, filterPoints=4, minArea=1000,
                                                contourType=myLib.CONTOURTS_POINTS_LINE)
    # imgTrans = img.copy()
    # imgObjectContour = img.copy()
    if len(resContours) != 0:
        # todo：将最大面积的物品作为背景
        biggestContour = resContours[0][2]
        imgContour, points = myLib.drawRect(imgContour, biggestContour, 2)
        imgTrans = myLib.transRectSelectedImg(img, biggestContour, wImg, hImg, pad)

        # todo: 在背景上识别卡片
        resObjectContours, imgObjectContour = myLib.getContours(imgTrans, filterPoints=4, minArea=200,
                                                                contourType=myLib.CONTOURTS_POINTS_LINE)
        for obj in resObjectContours:
            approxPoints = obj[2]

            points = myLib.reOrder(approxPoints)
            points = points.reshape((4,2))
            wInfo = round(myLib.distance(points[0]//scale, points[1]//scale) / 10)
            hInfo = round(myLib.distance(points[0] // scale, points[2] // scale) / 10)

            x,y,w,h = obj[3]

            cv2.putText(imgObjectContour, f"{wInfo}cm", (x+30, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv2.putText(imgObjectContour, f"{hInfo}cm", (x-70, y+h//2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)

        cv2.imshow("xxx", myLib.stackImages([imgTrans, imgObjectContour], scale))
        cv2.waitKey(0)

    cv2.imshow("xxx", myLib.stackImages([img, imgContour], scale))
    key = cv2.waitKey(1)
    if key == 27:
        break
