import time
from xml.etree.ElementTree import QName
import cv2
import numpy as np
import math
from math import sqrt, acos, degrees
import platform


def getCircle(center, inputImg, height, minRadius, maxRadius):

    condition = 0
    while condition < 20:
        condition += 1
        circles = cv2.HoughCircles(inputImg, cv2.HOUGH_GRADIENT, 2, height,
                                   param1=50, param2=100,
                                   minRadius=minRadius, maxRadius=maxRadius)

        if circles is not None and len(circles) == 1:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                circle = cv2.circle(
                    inputImg, center, radius, (255, 0, 255), 1)

            w0 = circle.shape[0]
            h0 = circle.shape[1]

            if len(circles) == 1:
                x, y, r = circles[0][0]
                mask = np.zeros((w0, h0), dtype=np.uint8)
                mask = cv2.circle(
                    mask, (x, y), r, (255, 255, 255), -1, 8, 0)
                return cv2.bitwise_and(inputImg, inputImg, mask=mask), center, True

        minRadius -= 10
        maxRadius -= 10
    return inputImg, center, False


def getEdgeDect(inputImg):
    return cv2.Canny(inputImg, 100, 400, None, 3)


def getLines(inputImg):
    return inputImg


def calcHypo(line):
    x1, y1, x2, y2 = line[0]

    dx = x2 - x1

    dx = negativeToPositive(dx)

    dy = y2 - y1

    dy = negativeToPositive(dy)

    return sqrt(dx ** 2 + dy ** 2)


def negativeToPositive(num):
    if(num < 0):
        return num * -1

    return num


def start(img):
    minRadius = 200 if platform.system() == 'Darwin' else 140
    maxRadius = 230 if platform.system() == 'Darwin' else 170
    height, width = img.shape
    center = 0
    watchFace, center, foundCircle = getCircle(
        center, img, height, minRadius, maxRadius)

    if not foundCircle:
        return img
    xcenter, ycenter = center
    edgedImg = getEdgeDect(watchFace)
    edgedImg = findLines(
        edgedImg, xcenter, ycenter, minRadius, maxRadius)
    return edgedImg


def findLines(edgedImg, xcenter, ycenter, minRadius, maxRadius):
    placeholder = cv2.cvtColor(edgedImg, cv2.COLOR_GRAY2BGR)
    bgrEdgedImage = np.copy(placeholder)
    x, y, w, h = cv2.boundingRect(edgedImg[0])

    lines = cv2.HoughLinesP(edgedImg, 1, np.pi / 180, 50, None, 40, 10)
    listOfLines = []
    xl1, xl2, yl1, yl2 = 0, 0, 0, 0
    xs1, xs2, ys1, ys2 = 0, 0, 0, 0
    if lines is not None:
        for line in lines:
            hypo = calcHypo(line)

            listOfLines.append(hypo)

        listOfLinesLength = len(listOfLines)
        listOfLines.sort(reverse=True)

        m = 0
        h = 0
        for f in range(listOfLinesLength):

            for line in lines:
                hypo2 = calcHypo(line)
                x1, y1, x2, y2 = line[0]

                if(hypo2 == listOfLines[0]):

                    m = hypo2
                    xl1 = x1
                    xl2 = x2
                    yl1 = y1
                    yl2 = y2

                    cv2.line(bgrEdgedImage, (xl1, yl1),
                             (xl2, yl2), (255, 0, 0), 3)

                if(m == listOfLines[0]):

                    if(hypo2 == listOfLines[f]):

                        if((sqrt((xl2 - x2)**2 + (yl2 - y2)**2)) > 20):

                            if((sqrt((xl1 - x1)**2 + (yl1 - y1)**2)) > 20):
                                xs1 = x1
                                xs2 = x2
                                ys1 = y1
                                ys2 = y2

                                cv2.line(bgrEdgedImage, (xs1, ys1),
                                         (xs2, ys2), (0, 255, 0), 3)
                                h = 1
                                break

            if(h == 1):
                break

        ahour, bhour = getAbs(xcenter, xs1, xs2)
        xhour, yhour = determineCoordinates(
            ahour, bhour, xs1, xs2, ys1, ys2)
        amin, bmin = getAbs(xcenter, xl1, xl2)
        xmin, ymin = determineCoordinates(amin, bmin, xl1, xl2, yl1, yl2)
        if not xhour + yhour == 0 or not xmin + ymin == 0:
            xycenter = (xcenter + ycenter) / 2

            p = (xhour + yhour) / 2
            z = (xmin + ymin) / 2
            q = 10

            centerRadius = 10

            if ((centerRadius + xycenter) > p and (xycenter - centerRadius) < p) and (((centerRadius + xycenter) > z and (xycenter - centerRadius) < z)):
                if (((p + q) > z and (p - q) < z)) or (((z + q) > p and (z - q) < p)):
                    ahour, bhour = getAbs(xcenter, xs1, xs2)
                    xhour, yhour = determineCoordinates(
                        bhour, ahour, xs1, xs2, ys1, ys2)
                    amin, bmin = getAbs(xcenter, xl1, xl2)
                    xmin, ymin = determineCoordinates(
                        bmin, amin, xl1, xl2, yl1, yl2)
                    p = (xhour + yhour) / 2
                    z = (xmin + ymin) / 2
                    calculateTime(xcenter, ycenter, xhour,
                                  yhour, xmin, ymin)
                    cv2.imshow('done', bgrEdgedImage)
                    cv2.waitKey(0)
                    exit()
        minRadius -= 5
        maxRadius -= 5
    return edgedImg


def getAbs(xcenter, x1, x2):
    x = abs(xcenter - x1)
    y = abs(xcenter - x2)
    return x, y

def determineCoordinates(x, y, xs1, xs2, ys1, ys2):
    if(x < y):
        x1 = xs1
        y1 = ys1
    else:
        x1 = xs2
        y1 = ys2
    return x1, y1

def calculateTime(xcenter, ycenter, xhour, yhour, xmin, ymin):
    l1 = sqrt(((xcenter - xhour) ** 2) + ((ycenter - yhour) ** 2))

    l2 = ycenter

    l3 = sqrt(((xcenter - xhour) ** 2) + ((0 - yhour) ** 2))

    cos_theta_hour = (((l1) ** 2) + ((l2) ** 2) -
                      ((l3) ** 2)) / (2 * (l1) * (l2))

    theta_hours_radian = acos(cos_theta_hour)

    theta_hours = math.degrees(theta_hours_radian)
    if(xhour > xcenter):
        right = 1
    else:
        right = 0

    if(right == 1):
        hour = round(theta_hours / (6*5))

    if(right == 0):
        hour = 12 - round(theta_hours / (6*5))

    if(hour == 0):
        hour = 12


    l1 = sqrt(((xcenter - xmin) ** 2) + ((ycenter - ymin) ** 2))

    l2 = ycenter

    l3 = sqrt(((xcenter - xmin) ** 2) + ((0 - ymin) ** 2))

    cos_theta_min = (((l1) ** 2) + ((l2) ** 2) -
                     ((l3) ** 2)) / (2 * (l1) * (l2))

    theta_min_radian = acos(cos_theta_min)

    theta_min = math.degrees(theta_min_radian)

    if(xmin > xcenter):
        right = 1
    else:
        right = 0

    if(right == 1):
        minute = round(theta_min / ((6*5)/5))

    if(right == 0):
        minute = 60 - (round(theta_min / ((6*5)/5)))
        if(xmin == xcenter):
            minute = 30 
        if(minute == 60):
            minute = 0
    print(f'Hour: {hour}')
    print(f'Minute: {minute}')


def getImage():
    vid = cv2.VideoCapture(0)

    while(True):
        ret, frame = vid.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edgedImg = start(grey)

        cv2.imshow('frame', edgedImg)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    vid.release()
    cv2.destroyAllWindows()


getImage()
