# image processing
# Author : Ji-yong
# Project Start:: 2021.04.01
# Last Modified from Ji-yong 2021.10.29


import cv2
import numpy as np
from numpy.core.numeric import zeros_like
from scipy.spatial import distance as dist
import numba
from tqdm import trange
import sys


# 색 리스트 반환 함수 (Minku koo)
@numba.jit(forceobj = True)
def createColorDict(image):
    colorDict = {}
    for y, row in enumerate(image):
        for x, bgr in enumerate(row):
            bgr = tuple(bgr)

            if colorDict == {}:
                colorDict[ bgr ] = [ (x, y) ]
            
            if bgr in colorDict.keys():
                colorDict[bgr].append( (x, y) )

            else:
                colorDict[bgr] = [ (x, y) ]
            
    return colorDict


# Contour 영역 내에 텍스트 쓰기
# https://github.com/bsdnoobz/opencv-code/blob/master/shape-detect.cpp
def setLabel(image, num, pt, radius):
    fontface = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.5 if radius / 100 < 0.5 else radius / 100 # 0.6
    thickness = 1 if scale < 0.8 else 2
    # thickness = 2 # 2

    textsize = cv2.getTextSize(num, fontface, scale, thickness)[0]
    pt = (int(pt[0]-(textsize[0]/2)+1), int(pt[1]+(textsize[1]/2)))

    cv2.putText(image, num, pt, fontface, scale, (150, 150, 150), thickness, 8)


# 컨투어 내부의 색을 평균내서 어느 색인지 체크
@numba.jit(forceobj = True)
def label(image, contour, lab, colorNames):
    mask = np.zeros(image.shape[:2], dtype="uint8")

    cv2.drawContours(mask, [contour], -1, 255, -1)

    mask = cv2.erode(mask, None, iterations=2)
    mean = cv2.mean(image, mask=mask)[:3]

    minDist = (np.inf, None)

    for (i, row) in enumerate(lab):
        d = dist.euclidean(row[0], mean)

        if d < minDist[0]:
            minDist = (d, i)
            
    return colorNames[minDist[1]]


# 해당 경로에서 이미지를 numpy형태로 반환
def getImageFromPath(path):
    return cv2.imread(path)


# 해당 이미지에서 색 추출
@numba.jit(forceobj = True)
def getColorFromImage(img):
    # 인식할 색 입력
    temp = [ (idx, color) for (idx, color) in enumerate(   list( createColorDict(img).keys() ),  1   ) ]

    return [str(i[0]) for i in temp], [i[1] for i in temp]


# 해당 이미지에서 contours, hierarchy, image_bin 반환
@numba.jit(forceobj = True)
def getContoursFromImage(img):
    # 이진화
    # cv2.COLOR_BGR2HSV

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    retval, image_bin = cv2.threshold(img, 127,255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return contours, hierarchy, image_bin


@numba.jit(forceobj = True)
def makeWhiteFromImage(img):
    return np.zeros(img.copy().shape) + 255


@numba.jit(forceobj = True)
def getImgLabelFromImage(colors, img):
    lab = np.zeros((len(colors), 1, 3), dtype="uint8")
    for i in range(len(colors)):
        lab[i] = colors[i]

    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)

    # 색검출할 색공간으로 LAB사용
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    return img_lab, lab


# @numba.jit(forceobj = True)
def getRadiusCenterCircle(raw_dist):
    dist_transform, label = cv2.distanceTransformWithLabels(raw_dist, cv2.DIST_L2, maskSize=5)

    _, radius, _, center = cv2.minMaxLoc(dist_transform)

    points = None

    # 넘버링 여러 곳에 하는 기능 개발 중, 주석처리하였음

    # dist_transform = cv2.distanceTransform(raw_dist, cv2.DIST_L2, maskSize=5)
    # points = [list(dist_transform)[i] for i in range(0, len(dist_transform), 300) if list(dist_transform)[i] > 50]
    # print(type(list(dist_transform)[50]), list(dist_transform)[50])

    # ret, dist1 = cv2.threshold(dist_transform, 0.6*dist_transform.max(), 255, 0)
    

    # print(f'dist_transform : {dist_transform}')
    # print(f'label: {label}')

    # points = []
    # points = np.where(dist_transform > 10)



    # for idx in range(0, len(dist_transform)-30, 30):
    #     _, radius, _, center = cv2.minMaxLoc(dist_transform[idx:idx+30])
    #     if radius > 10:
    #         points.append((radius, center))

    # print(np.unique(np.where(label > 10)))


    # result = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # minVal, maxVal, a, center = cv2.minMaxLoc(result)

    return radius, center, points



# @numba.jit(forceobj = True)
def setColorNumberFromContours(img, thresh, contours, hierarchy, img_lab, lab, colorNames, gif_mode):

    cnt = 9
    # 컨투어 리스트 Looping
    for idx in trange(len(contours), file=sys.stdout, desc='Set Numbering'):
        contour = contours[idx]

        # 면적 
        if cv2.contourArea(contour) < 80: continue

        chlidren = [ i for i, ii in enumerate(hierarchy[0]) if ii[3] == idx ]

        raw_dist = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(raw_dist, contour, -1, (255, 255, 255), 1)
        cv2.fillPoly(raw_dist, pts =[contour], color=(255, 255, 255))
        cv2.fillPoly(raw_dist, pts =[contours[i] for i in chlidren], color=(0, 0, 0))


        # 내접원 반지름, 중심좌표 추출
        radius, center, points = getRadiusCenterCircle(raw_dist)

        # 반지름 작은거 무시
        if radius < 8: continue

        if center is not None:
            # 넘버링 여러 곳에 하는 기능 개발 중, 주석처리 하였음
            # for radius, center in points:

            cv2.drawContours(img, [contour], -1, (150, 150, 150), 1)

            # 내접원 확인용(주석 풀면 활성화)
            # cv2.circle(img, center, int(radius), (0, 255, 0), 1, cv2.LINE_8, 0)

            # 컨투어 내부에 검출된 색을 표시
            color_text = label(img_lab, contour, lab, colorNames)

            center_ = (center[0], center[1])
            setLabel(img, color_text, center_, radius)
            if gif_mode:
                cv2.imwrite(f'D:/ppt_img/img{str(cnt).zfill(4)}.png', img)
            cnt += 1

    return img

    
@numba.jit(forceobj = True)
def setColorLabel(img, colorNames, colors):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1 # 0.6
    thickness = 2 # 2

    for idx in range(len(colors)):
        cv2.putText(img, colorNames[idx], (20, 40*(idx+1)), fontface, scale, (50, 50, 50), thickness, 8)
        cv2.rectangle(img, (60, 40*(idx+1)-20), (90, 40*(idx+1)), tuple([int(i) for i in colors[idx]]), -1, 8)
        print(colors[idx])

    return img