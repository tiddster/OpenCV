import pytesseract
import cv2
import numpy as np

image = cv2.imread('444.png')

# image2string
def ORC():
    text = pytesseract.image_to_string(image)
    print(text)
    return text

# 将每个识别到的字符都加上边框范围，返回的是(x,y,w,h)的多个元组
def detect_boxes():
    pytesseract.image_to_boxes(image)

# 为每一个字符加上红色边框框
def add_boxes():
    hImg, wImg, _ = image.shape
    boxes = pytesseract.image_to_boxes(image)
    for b in boxes.splitlines():
        b = b.split(' ')
        x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(image, (x,hImg-y), (w, hImg-h), (0,0,255), 1)
        cv2.putText(image,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

# 检测整个单词而非单个字符
def detect_words():
    hImg, wImg, _ = image.shape
    datas = pytesseract.image_to_data(image)
    for d in datas.splitlines()[1:]:
        d = d.split()
        if len(d) == 12:
            x, y, w, h = int(d[6]), int(d[7]), int(d[8]), int(d[9])
            cv2.rectangle(image, (x, y), (w+x,h+y), (0, 0, 255), 1)
            cv2.putText(image, d[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

def detect_only_num():
    cong = r'--oem 3 --psm 6 outputbase digits'
    datas = pytesseract.image_to_data(image, config=cong)
    for d in datas.splitlines()[1:]:
        d = d.split()
        if len(d) == 12:
            x, y, w, h = int(d[6]), int(d[7]), int(d[8]), int(d[9])
            cv2.rectangle(image, (x, y), (w+x,h+y), (0, 0, 255), 1)
            cv2.putText(image, d[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

detect_only_num()
cv2.imshow('result', image)
cv2.waitKey(0)
