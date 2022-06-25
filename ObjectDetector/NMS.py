import torch

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

def nms(bboxes, confidence, threshold = 0.5):
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
            print(aimIndex, i)
            if IOU(aimBox, boxes[i]) > threshold:
                indexes.remove(i)
        indexes.remove(aimIndex)
        resIndex.append(aimIndex)
    return resIndex