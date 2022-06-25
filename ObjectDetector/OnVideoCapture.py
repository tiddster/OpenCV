import cv2
import ObjectDetector as OD

capture = cv2.VideoCapture(0)

net, names = OD.objectDetector()

while True:
    success, img = capture.read()

    classIDs, confidence, bbox = net.detect(img, confThreshold=0.5)
    print(classIDs, confidence, bbox)

    if len(classIDs) != 0:
        for ID, conf, box in zip(classIDs, confidence, bbox):
            str_conf = "{:.2f}".format(conf)
            cv2.rectangle(img, box, color=(0, 0, 255), thickness=3)
            cv2.putText(img, names[ID - 1] + " " + str_conf, (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255))

    cv2.imshow("OUTPUT", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
