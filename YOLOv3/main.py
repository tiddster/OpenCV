import cv2
import numpy as np
import yolov3Model
from MyOpenCVLib import myLib

capture = cv2.VideoCapture(0)

whT = 320

net = yolov3Model.train_net()
classNames = yolov3Model.classNames

while True:
    success, img = capture.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0], 1, crop = False)
    net.setInput(blob)

    # 获取每一卷积层的名称
    layerNames = net.getLayerNames()
    # 获取输出层的索引
    outLayerIndex = net.getUnconnectedOutLayers()
    outputNames = [layerNames[index - 1] for index in outLayerIndex]

    """
    利用FP算法得到三个输出层
    三个输出层的维度分别是(300, 85), (1200, 85), (4800, 85)
    维度一：将图像分为300、1200、4800个单元格，对每一个单元格进行检测
    维度二：5+80：
        5：(中心点x，中心点y，宽，高，该单元个中含有物品的置信度)
        80：coco集中被标记的80个物品，每一个物品的预测概率值
    """
    outputs = net.forward(outputNames)

    bbox, classIds, confs = yolov3Model.findObjects(outputs, img)
    resName = [classNames[i] for i in classIds]

    indexes1 = cv2.dnn.NMSBoxes(bbox, confs, 0.5, nms_threshold=0.5)
    indexes2 = myLib.nms(bbox, confs)

    for i in indexes1:
        index = classIds[i]
        str_conf = " {:2.0f}%".format(confs[i] * 100)
        cv2.rectangle(img, bbox[i], (0, 0, 255), 2)
        cv2.putText(img, classNames[index] + f"{str_conf}", (bbox[i][0], bbox[i][1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    cv2.imshow("xxx", img)
    key = cv2.waitKey(1)
    if key == 27:
        break