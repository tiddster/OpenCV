import cv2
import ObjectDetector as OD

img = cv2.imread('dataset\\pic\\4.jpg')

net, names = OD.objectDetector()

# 检测一个图片的名称id，置信度，边框
classIDs, confidence, bbox = net.detect(img, confThreshold=0.5)

for ID, conf, box in zip(classIDs, confidence, bbox):
    cv2.rectangle(img, box, color=(0,0,255),thickness=3)
    cv2.putText(img, names[ID-1] , (box[0]+10, box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))

cv2.imshow("img", img)
cv2.waitKey(0)