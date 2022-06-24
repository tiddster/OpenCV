import cv2
import math

path = '111.png'
img = cv2.imread(path)
points = []
hImg, wImg, _ = img.shape

# 鼠标点击事件
def mousePoint(event, x, y, flage, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (0, 0, 255), cv2.FILLED)
        points.append([x,y])
        if len(points) % 3 == 2:
            plt1, plt2 = points[-2:]
            plotLines(plt1, plt2)
        elif len(points) % 3 == 0 and len(points) != 0:
            plt1, plt2, plt3 = points[-3:]
            plotLines(plt2, plt3)

def plotLines(plt1, plt2):
    cv2.line(img, plt1, plt2, (0,0,255))

def calAngle(plt1, plt2, plt3):
    a = ((plt1[0] - plt2[0]) ** 2 + (plt1[1] - plt2[1])**2) ** 0.5
    b = ((plt1[0]- plt3[0])  ** 2 + (plt1[1] - plt3[1])**2) ** 0.5
    c = ((plt2[0] - plt3[0]) ** 2 + (plt2[1] - plt3[1]) ** 2) ** 0.5
    cosb = (-b**2 + c**2 + a**2) / (2*a*c)
    return cosb

def getAngle(pointsList):
    plt1, plt2, plt3 = pointsList[-3:]
    cos = calAngle(plt1, plt2, plt3)
    angR = math.acos(cos)
    angD = round(math.degrees(angR))
    cv2.putText(img, str(angD), (plt2[0] - 20, plt2[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    print(angD)

while True:
    if len(points) % 3 == 0 and len(points) != 0:
        getAngle(points)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mousePoint)
    key = cv2.waitKey(1)
    if key == ord('r'):
        points = []
        img = cv2.imread(path)
    elif key == 27:
        break
