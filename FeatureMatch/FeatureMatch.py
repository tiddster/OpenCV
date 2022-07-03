import cv2
import numpy as np
import myLib


# 配置检测器
def getOrbDetector(featureNums=1000):
    orb = cv2.ORB_create(featureNums)
    return orb


# 利用检测器得到图像特征和相关描述
def getKeyPointsAndDescription(orb, img):
    keyPs, des = orb.detectAndCompute(img, None)
    return keyPs, des


# 第一个是目标的描述，第二个是判断是否与目标匹配的图像描述
def getGoodMatch(desTarget, desTobeMatched):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desTarget, desTobeMatched, k=2)
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatches.append(m)
    return goodMatches


def getHomography(keyPtsTarget, keyTobeMatched, goodMatches):
    """
    对于tobeMatched的图片，无论从那个方向看，总是存在相对于Target图片位置相同的点
    通过这些位置相同的点构造变换矩阵的映射， 即可将TobeMatched图片位置，方向，大小转换为Target图片的位置，方向，大小
    :param keyPtsTarget:  Target图片关键特征点
    :param keyTobeMatched:  Matched图片关键特征点
    :param goodMatches: 两个图片匹配成功的点
    :return: 映射矩阵，特征掩盖（就算把match图片掩盖一点，也可以对算出映射矩阵）
    """

    srcPts = np.float32([keyPtsTarget[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    dstPts = np.float32([keyTobeMatched[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
    return matrix, mask