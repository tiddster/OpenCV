import cv2
import numpy as np
import FeatureMatch
import myLib

#####################################
imgTarget = cv2.imread('111.jpg')
video = cv2.VideoCapture('F:\\OpenCV-SP\\AROpenCV\\555.mp4')
capture = cv2.VideoCapture(0)
wImg, hImg = 320, 640
#####################################

imgTarget = cv2.resize(imgTarget, (wImg, hImg))
orb = FeatureMatch.getOrbDetector()
keyPtsTarget, desTarget = FeatureMatch.getKeyPointsAndDescription(orb, imgTarget)
# 可以画出检测到的关键点
# imgTarget = cv2.drawKeypoints(imgTarget, keyPsTarget, None)

while True:
    # success, videoFrame = video.read()
    # videoFrame = cv2.resize(videoFrame, (wImg, hImg))
    # cv2.imshow('xx', videoFrame)

    success, imgCap = capture.read()
    #imgCap = cv2.imread('222.png')
    orb = FeatureMatch.getOrbDetector()
    keyPtsCap, desCap = orb.detectAndCompute(imgCap, None)
    imgCap = cv2.drawKeypoints(imgCap, keyPtsCap, None)
    cv2.imshow("xxx", imgCap)

    if desCap is not None:
        goodMatches = FeatureMatch.getGoodMatch(desTarget, desCap)
        imgFeature = cv2.drawMatches(imgTarget, keyPtsTarget, imgCap, keyPtsCap, goodMatches, None, flags=2)
        print("READY")
        if len(goodMatches) > 50:
            success, videoFrame = video.read()
            matrix, mask = FeatureMatch.getHomography(keyPtsTarget, keyPtsCap, goodMatches)

            # 将目标图在匹配图中框出来， ptsCapRect为匹配图中的框代表的四个点
            pts = np.float32([[0,0], [0, hImg], [wImg, hImg], [wImg, 0]]).reshape(-1,1,2)
            ptsCap = cv2.perspectiveTransform(pts, matrix)
            ptsCap = np.int32(ptsCap)
            x, y, w, h = myLib.getXYWHAccordingTo4Points(ptsCap)
            ptsCapRect = [[x, y], [x, y+h], [x+w, y+h], [x+w, y]]
            imgWithRect = cv2.polylines(imgCap, [np.array(ptsCapRect)], True, (0, 255, 0), 3)
            imgCap = cv2.resize(imgCap, (imgWithRect.shape[1], imgWithRect.shape[0]))

            videoFrame = cv2.resize(videoFrame, (w, h))
            ptsVideoFrame = np.float32([[0, 0], [0, h], [w,h], [w,0]])
            matrix_VideoFrame2ImgWithRect = cv2.getPerspectiveTransform(ptsVideoFrame, np.float32(ptsCapRect))
            videoFrameTrans = cv2.warpPerspective(videoFrame, matrix_VideoFrame2ImgWithRect, (imgWithRect.shape[1], imgWithRect.shape[0]))

            maskBG = np.zeros_like(imgWithRect)
            maskBG = cv2.fillPoly(maskBG, [np.array(ptsCapRect)], (255, 255, 255))

            maskBG = cv2.bitwise_not(maskBG)
            imgCap = cv2.bitwise_and(maskBG, imgCap)
            videoFrameRes = cv2.bitwise_or(imgCap, videoFrameTrans)

            cv2.imshow("xxx", videoFrameRes)
    key = cv2.waitKey(25)
    if key == 27:
        break
