import cv2
import numpy as np
import utils

capture = cv2.VideoCapture(0)
wImg, hImg = 480, 640

# 初始化阈值调整框
utils.initializeTrackbars()

while True:
    success, img = capture.read()
    # 图像处理中三种最基础的图像增强：原画，灰度化，高斯滤波
    img = cv2.resize(img, (wImg, hImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(img, (5,5), 1)

    # 利用阈值调整框获取两个阈值，利用Canny读取图像的边缘特征
    thres = utils.valTrackbars()
    imgThres = cv2.Canny(imgBlur,thres[0], thres[1])

    # 将边缘特征放大化处理
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgThres, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    # 利用边缘特征图像绘制图像边缘
    imgContours = img.copy()
    imgFourPointsContours = img.copy()
    contours, hierarchy =cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0,255,0),10)

    # 利用边缘图像特征找到面积最大的轮廓
    biggest, maxArea = utils.biggestContour(contours)
    if biggest.size != 0:
        # 将最大的轮廓中边缘四点标注出来
        biggest = utils.reorder(biggest)
        cv2.drawContours(imgFourPointsContours, biggest, -1, (0,255,0), 20)
        imgFourPointsContours = utils.drawRectangle(imgFourPointsContours, biggest, 2)

        # 按照图像边缘四个点做转化，只呈现由边缘四个点以及边缘框出来的图像
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0], [wImg, 0], [0, hImg], [wImg, hImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgTrans = cv2.warpPerspective(img, matrix, (wImg, hImg))
        imgArray = [
            [img, imgGray, imgThres],
            [imgContours,imgFourPointsContours, imgTrans]
        ]
    else:
        imgArray = [
            [img, imgGray, imgThres],
            [imgContours, imgFourPointsContours, img]
        ]

    labels = [
        ["Ori", "Gray", "Thres"],
        ["Contours", "FourPC", "Trans"]
    ]
    cv2.imshow("sss", utils.stackImages(imgArray, 0.75, labels))
    key = cv2.waitKey(1)
    if key == 27: break
