import cv2
import numpy as np
import face_recognition
from FaceRecognition.FRnet import *

imgElon = face_recognition.load_image_file('ImageBasic/Elon Mask.png')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/Elon Test.png')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


encodeElon = faceLocAndEncode(imgElon)
encodeTest = faceLocAndEncode(imgTest)

res = compare_faces([encodeElon],encodeTest)
faceDis = face_distance([encodeElon],encodeTest)
print(res, faceDis)

cv2.imshow("Elon Mask", imgElon)
cv2.waitKey(0)
cv2.imshow("Elon Test", imgTest)
cv2.waitKey(0)