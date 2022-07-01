import cv2
import numpy as np
import dlib
import myLib

predictor_weight_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_weight_path)


def createBox(img, points,scale=1, text="none"):

    mask = np.zeros_like(img)
    # 根据特征点将嘴唇轮廓用多边形拟合出来，并且填充
    mask = cv2.fillPoly(mask,  [points], (255,255,255))
    #imgMask = cv2.bitwise_and(img, mask)

    bbox = cv2.boundingRect(points)
    x, y, w, h = bbox
    #cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255),2)
    #cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
    imgCut = img[y:y+h, x:x+w]
    imgCut = cv2.resize(imgCut, (0,0), None, scale, scale)
    return imgCut, mask


def getFacialDetector(imgGray, img):
    faces = detector(imgGray)
    imgMark = img.copy()

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        imgMark = cv2.rectangle(imgMark, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(imgGray, face)
        points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append([x,y])
            #cv2.circle(imgMark, (x, y), 5, (50, 50, 255), cv2.FILLED)
            #cv2.putText(imgMark, str(i), (x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255))
        #imgLeftEye = createBox(imgMark, np.array(points[36:42]), text="LeftEye")
        #imgRightEye = createBox(imgMark, np.array(points[42:48]), text="RightEye")
        #imgNose = createBox(imgMark, np.array(points[27:36]), text="Nose")
        imgMouth, imgMouthMask = createBox(imgMark, np.array(points[48:]), text="Mouth")
        #cv2.imshow('xx', imgMouth)
        cv2.imshow("xxx", imgMouthMask)

        imgColorMouth = np.zeros_like(imgMouthMask)
        imgColorMouth[:] = 153,0,157
        imgColorMouth = cv2.bitwise_and(imgMouthMask, imgColorMouth)
        imgColorMouth = cv2.GaussianBlur(imgColorMouth, (7,7), 10)
        imgColorMouth = cv2.addWeighted(img, 1, imgColorMouth, 0.4, 0)
        cv2.imshow("xxxx", imgColorMouth)

    return imgMark
