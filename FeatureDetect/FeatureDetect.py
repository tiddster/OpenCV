import cv2
import numpy as np
from tqdm import tqdm


def getORB(featureNum=1000):
    orb = cv2.ORB_create(featureNum)
    return orb


def train_KeyPointsAndDescriptor(orb, images):
    """
    keyPoints: 特征点及其位置
    descriptor: 用32个值描述每个特征点
    :param orb:
    :param images:
    :return:
    """
    keyPointsList = []
    descriptorList = []
    for image in tqdm(images):
        keyPts, des = orb.detectAndCompute(image, None)
        keyPointsList.append(keyPts)
        descriptorList.append(des)
    return keyPointsList, descriptorList


def findBestMatches(desAim, desList, minThres = 50):
    """
    利用k邻近算法计算特征点descriptor的匹配程度
    匹配度最高的做为匹配项
    返回匹配项的索引和匹配项与目标的匹配矩阵
    :param desAim:  匹配目标的descriptor
    :param desList:    descriptorList
    :param minThres:  最小匹配阈值
    :return:  返回匹配项的索引和匹配项与目标的匹配矩阵
    """
    bf = cv2.BFMatcher()
    bestMatches = []
    bestIndex = -1
    for index, des in enumerate(desList):
        goodMatches = []
        matches = bf.knnMatch(des, desAim, k=2)

        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatches.append([m])

        if len(goodMatches) > len(bestMatches) and len(goodMatches) > minThres:
            bestMatches = goodMatches
            bestIndex = index

    return bestIndex, bestMatches
