import cv2
import numpy as np
import AROpenCV
import myLib

capture = cv2.VideoCapture(0)
imgTarget = cv2.imread("333.jpg")
videoFile = cv2.VideoCapture('555.mp4')
wImg, hImg = 640, 480
# AROpenCV.RectCapture(imgTarget)

while True:
    success, image = capture.read()
    image = cv2.resize(image, (wImg, hImg))
    successBBox, biggestBBox, approxPoints = AROpenCV.RectCapture(image)

    if successBBox:
        x, y, w, h = biggestBBox
        success, video = videoFile.read()
        video = cv2.resize(video, (w,h))

        cv2. rectangle(image, biggestBBox, (0,255,0),2)
        points = [[x,y], [x+w, y], [x, y+h], [x+w, y+h]]

        # 映射转换！！！
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
        imgTrans = cv2.warpPerspective(video, matrix, (wImg, hImg))

        points = [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]
        mask = AROpenCV.fillContour(biggestBBox, np.array(points), image)
        mask = cv2.bitwise_not(mask)

        image = cv2.bitwise_and(mask, image)
        imgRes = cv2.bitwise_or(image, imgTrans)

        cv2.imshow("sss", myLib.stackImages([imgTrans, image, imgRes]))
        cv2.waitKey(1)