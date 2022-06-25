import cv2
import ObjectDetector as OD
import NMS

capture = cv2.VideoCapture(0)
# 加入nms（非极大值抑制），目的是去除冗余的边框
nms_threshold = 0.5

net, names = OD.objectDetector()

while True:
    success, img = capture.read()

    classIDs, confidence, bbox = net.detect(img, confThreshold=0.5)

    # 利用nms在boxes中去掉冗杂项，只保留极大值项对应的索引
    # 上面是调库，下面手写，对比一下效果
    indices1 = cv2.dnn.NMSBoxes(bbox, confidence, 0.5, nms_threshold=nms_threshold)
    indices2 = NMS.nms(bbox, confidence, nms_threshold)
    print(indices1, indices2)

    for i in indices2:
        box = bbox[i]
        cv2.rectangle(img, box, color=(0, 0, 255), thickness=3)
        cv2.putText(img, names[classIDs[i] - 1] + " {:2.0f}%".format(confidence[i] * 100), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255))

    # if len(classIDs) != 0:
    #     for ID, conf, box in zip(classIDs, confidence, bbox):
    #         cv2.rectangle(img, box, color=(0, 0, 255), thickness=3)
    #         cv2.putText(img, names[ID - 1] + " {:2.0f}%".format(conf*100), (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255))

    cv2.imshow("OUTPUT", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
