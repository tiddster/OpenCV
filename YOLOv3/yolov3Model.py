import cv2
import numpy as np
from numba import jit

path = "F:\\YOLOv3Dataset\\coco.names"
classNames = []

with open(path, 'rt') as f:
    classNames = f.read().split('\n')

modelCfg_path = 'F:\\YOLOv3Dataset\\yolov3-320.cfg'
modelWeight_path = 'F:\\YOLOv3Dataset\\yolov3-320.weights'


def train_net():
    # 加载Darknet网络
    net = cv2.dnn.readNetFromDarknet(modelCfg_path, modelWeight_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # 利用CPU运行
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

def findObjects(outputs, img, threshold=0.5):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int(detection[0] * wT - w / 2), int(detection[1] * hT - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(confidence)
    return bbox, classIds, confs
