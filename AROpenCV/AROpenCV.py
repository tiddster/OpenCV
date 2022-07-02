import cv2
import numpy as np

scale = 0.5

def RectCapture(img):
    # img = cv2.resize(img, (int(w * scale), int(h * scale)))

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(img, (5,5), 10)

    imgCanny = cv2.Canny(imgBlur, 100, 200)

    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel)
    imgThres = cv2.erode(imgDial, kernel)

    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggestBBox = []
    approxPoints = []
    maxArea = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > maxArea:
            peri = cv2.arcLength(contour, True)
            approxPoints = cv2.approxPolyDP(contour, 0.02*peri, True)
            biggestBBox = cv2.boundingRect(approxPoints)

    if len(biggestBBox) != 0:
        return True, biggestBBox, approxPoints
    else:
        return False, biggestBBox, approxPoints

def fillContour(bbox, approxPoints, img):
    mask = np.zeros_like(img)
    # 根据特征点将嘴唇轮廓用多边形拟合出来，并且填充
    mask = cv2.fillPoly(mask, [approxPoints], (255,255,255))
    # imgMask = cv2.bitwise_and(img, mask)
    return mask