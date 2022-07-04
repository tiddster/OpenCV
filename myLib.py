import cv2
import numpy as np
import os
"""
读取文件夹所有图片以及图片的名字
"""
def readImagesAndNamesFromLib(dirPath, resizes = []):
    """
    :param dirPath:   文件夹的名字
    :param resizes:     可以设置读取图像的宽、高，默认不设置
    :return:
    """
    imagesLib = os.listdir(dirPath)
    images = []
    names = []
    for imageName in imagesLib:
        image = cv2.imread(f'{dirPath}\\{imageName}')
        if len(resizes) != 0:
            image = cv2.resize(image, (resizes[0], resizes[1]))
        images.append(image)
        names.append(imageName.split('.')[0])
    return images, names


"""
用于opencv一个窗口展示多个图像
输入：图像的组合数组，规模，对应标签
输出：一整个合并图像，可以直接用cv2.imshow("xx",xx)输出
"""


def stackImages(imgArray, scale=0.5, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d], (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 255), 2)
    return ver


"""
手动实现nms算法，目标检测中的基础算法
目的：在目标检测时，对于同一个物体可能会检测出很多中情况，检测边框会发生重叠
            所以需要去掉冗余的边框，找到冗余边框中最大的那个
方法：利用IOU，即交并比，通过判断多个边框的重合面积判断其是否冗余
代码：有ABCDEF boxes,
            先按照置信度进行排序，排序为FCBEDA
            从中取出F，与剩下进行IOU比较，若IOU过大则去掉某些Boxes
            BD与F的IOU过大，BD去掉，将F加入结果中，还剩CEA
            从中取出C，与剩下的进行IOUT比较.....
            重复此过程直到boxes中没有元素可比
            返回结果
"""
"""
box = x,y,w,h
"""


def IOU(bbox1, bbox2):
    if bbox1[0] < bbox2[0]:
        box1 = bbox1
        box2 = bbox2
    else:
        box1 = bbox2
        box2 = bbox1
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ox, oy, ow, oh = x2, y2, x1 + w1 - x2, y1 + h1 - y2

    S1, S2, oS = w1 * h1, w2 * h2, ow * oh
    iou = oS / (S1 + S2 - oS)
    return iou


def nms(bboxes, confidence, threshold=0.5):
    resIndex = []
    boxes = list(bboxes).copy()

    # 将置信值放入字典，以让索引值和置信值一一对应，并按照大小排序
    conf_dict = {}
    for i in range(len(confidence)):
        conf_dict.update({i: confidence[i]})
    indexes = sorted(conf_dict, reverse=True)

    while len(indexes) != 0:
        aimIndex = indexes[0]
        aimBox = boxes[aimIndex]
        for i in indexes[1:]:
            if IOU(aimBox, boxes[i]) > threshold:
                indexes.remove(i)
        indexes.remove(aimIndex)
        resIndex.append(aimIndex)
    return resIndex


"""
 图像边缘特征提取
 步骤：
    1. 过滤去噪
    2. 双阈值抑制提取边缘
    3. 强化边缘特征
    4. 根据面积筛选特征区域，获取近似多边形顶点和最小边缘矩形的(x,y,w,h)
"""
CONTOURTS_NONE = 0
CONTOURTS_LINE = 1
CONTOURTS_POINTS_LINE = 2


def getContours(img, thres=[100, 200], contourType=CONTOURTS_LINE, minArea=1000, filterPoints=0):
    """
    :param img:  原始图像
    :param thres:  双阈值
    :param contourType:  绘制图像边缘类型：不绘制、绘制特征边、绘制近似多边形顶点连起来的边
    :param minArea:  最小面积阈值
    :param filterPoints:  期望多边形顶点数
    :return: 边缘，绘制了边缘的图像
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    # 双阈值抑制， 抑制因噪声或其他产生的虚边缘
    imgCanny = cv2.Canny(imgBlur, thres[0], thres[1])

    # 将边缘特征放大化处理
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    # 获取边缘
    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resContours = []

    # 根据面积筛选特征边缘
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            # 计算轮廓周长
            peri = cv2.arcLength(contour, True)
            # 利用多边形近似边缘，返回近似多边形的顶点
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            # 得到近似多边形的最小边缘矩形
            bbox = cv2.boundingRect(approx)

            # 根据近似多边形的顶点数判断是否是我们需要的多边形
            # 利用filterPoints设置我们期望的多边形顶点数，例如我们需要矩形就设置filterPoints = 4
            # 如果没有设置则默认没有期望的多边形，直接将所有近似多边形加入结果中，否则之加入对应多边形
            if filterPoints > 0:
                if len(approx) == filterPoints:
                    resContours.append((len(approx), area, approx, bbox, contour))
            else:
                # 将（近似多边形的顶点数，面积，近似多边形顶点，最小边缘矩形顶点，边缘特征）作为元组加入结果
                resContours.append((len(approx), area, approx, bbox, contour))

    resContours = sorted(resContours, key=lambda x: x[1], reverse=True)

    imgContours = img.copy()
    if contourType != CONTOURTS_NONE and len(resContours) != 0:
        for contourTulpe in resContours:
            if contourType == CONTOURTS_LINE:
                imgContours = cv2.drawContours(imgContours, contourTulpe[4], -1, (0, 255, 0), 10)
            elif contourType == CONTOURTS_POINTS_LINE:
                imgContours = cv2.drawContours(imgContours, contourTulpe[2], -1, (0, 255, 0), 20)
                imgApprox = drawRect(imgContours, contourTulpe[2], 3)
    return resContours, imgContours


"""
因为读取的四边形四个点不确定，顺序不总是一样的，所以需要先将其整理统一
判断四个角的顺序：
1   2
3   4
"""
def reOrder(approxPoints):
    """
    :param approxPoints: 多边形的近似顶点 (由函数getContours得到)
    :return:
    """
    newPoints = np.zeros_like(approxPoints)
    points = approxPoints.reshape((4, 2))
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints.reshape((4, 2))


"""
根据四边形的顶点, 连接四边形
"""
def drawRect(img, approxPoints, thickness):
    """
    :param img:  原始图像
    :param approxPoints:  多边形的近似顶点
    :param thickness:  线宽
    :return:
    """
    points = approxPoints.reshape((4,2))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], (0, 255, 0), thickness)
    cv2.line(img, points[-1], points[0], (0, 255, 0), thickness)
    return img, points


"""
按照points选中的矩形裁剪图片，并输出成一个图像
"""
def transRectSelectedImg(img, approxPoints, w, h, pad = 20):
    """
    :param img: 需要转换的图像
    :param approxPoints:  多边形的近似顶点
    :param w:  转换后图像的宽
    :param h:   转换后图像的高
    :param pad:     边缘噪声裁剪
    :return:
    """
    points = reOrder(approxPoints)
    points = points.reshape((4, 2))
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgTrans = cv2.warpPerspective(img, matrix, (w, h))
    imgTrans = imgTrans[pad:imgTrans.shape[0]-pad, pad:imgTrans.shape[1]- pad]

    return imgTrans


def distance(plt1, plt2):
    return ((plt2[0] - plt1[0]) ** 2 + (plt2[1] - plt1[1]) ** 2) ** 0.5


def getXYWHAccordingTo4Points(points):
    """
    根据矩阵四个点算出宽高
    先用mylib中函数将points重新排序为：
    1    2
    3   4
    再进行计算
    :param points:  矩阵的四个点
    :return: 宽和高
    """
    points = reOrder(points)
    x1, x2, x3, x4 = points[:, 0]
    w = (x2 + x4 - x3 - x1) // 2
    y1, y2, y3, y4 = points[:, 1]
    h = (y3 + y4 - y2 - y1) // 2
    return x1,  y1, w,  h