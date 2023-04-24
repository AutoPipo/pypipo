﻿import cv2
import os

import numpy as np
from libs.colorCode import HexColorCode

class Painting:
    def __init__(self, imagepath):
        # K-Means 알고리즘 이용한 색상 군집화된 이미지
        self.np_color_clustered_img = np.array([])
        # 지정된 색상과 매칭된 이미지
        self.np_color_matched_img = np.array([])
        
        self.original_img = cv2.imread(imagepath) # Original Image
        # self.file_base_name = os.path.basename(imagepath) # file base name
        # self.filename = self.file_base_name.split(".")[0] # file name
        
        # 지정된 hex color 리스트
        self.hexColorCode =  HexColorCode().hexColorCodeList
        self.colorName = HexColorCode().colorNameList
        
        # clustered color list
        self.np_clustered_colors = np.array([])
        self.list_clustered_color_names = []
    
    # image blurring
    def blurring(self, 
                div = 8, 
                radius = 10, 
                sigmaColor = 20, 
                medianValue = 5,
                step = 0) :
        """
        Parameters
            div <int> : Reducing numbers of color on Image (default = 8)
            radius <int> : bilateralFilter Parameter (default = 10)
            sigmaColor <int> : bilateralFilter Parameter (default = 20)
            medianValue <int> : medianBlur Parameter (default = 5)
            step <int> : Blurring intensity by step size (0<=step<=5, default = 0)
        returns
            blurring <np.ndarray> : blurred Image
        """
        
        qimg = self.original_img.copy() # copy original image
        
        step = min(max(0, step), 5) # 1<= step <= 5
        
        imageSize = int( (qimg.shape[1] * qimg.shape[0]) ** 0.5 ) // 100
        # set sigmaColor, radius by imageSize and step
        sigmaColor += min( int(imageSize * 2.5) , 90) + step * 4
        radius += min( int(imageSize * 1.5) , 40) + step * 2
        
        # blurring
        blurring = cv2.bilateralFilter(qimg, radius, sigmaColor, 60)
        blurring = cv2.medianBlur(blurring, medianValue)
        
        # reduce numbers of color
        blurring = blurring // div * div + div // 2
        
        return blurring
   
    # color clustering
    def colorClustering(self, image, cluster = 16, round = 1): 
        self.np_color_clustered_img, sse = self.__kmeansColorCluster(image, 
                                                                    clusters = cluster, 
                                                                    rounds = round)
        return self.np_color_clustered_img
   
    # 여기에 확장한 이미지랑 클러스터 칼라 매칭 
    def expandImageColorMatch(self, expand_image):
        self.np_color_clustered_img =  self.__matchColors(expand_image, self.np_clustered_colors)
        return self.np_color_clustered_img
    
    # color on image match with specified hex colors
   # hexColorCode와 매칭하면 이상한 색상이 추출됨 -> 사용 보류
    def getPaintingColorMap(self, clustered_image):
        hexColors = np.array([ self.__hex2bgr(hex) for hex in self.hexColorCode ])
        self.np_color_matched_img = self.__matchColors(clustered_image, hexColors)
        return self.np_color_matched_img
    
    # counting numbers of color
    def getNumberOfColor(self, image):
        """
        Parameters
            image <np.ndarray> : image
        returns
            numbers of color on image <int>
        """
        colorDict = {} # Key : Color Code / Values : Pixel Position
        for y, row in enumerate(image):
            for x, bgr in enumerate(row):
                bgr = tuple(bgr) # np.ndarray convert to tuple
                if colorDict == {}: # if dictionary in empty
                    colorDict[ bgr ] = [ (y, x) ]
                    continue
                
                if bgr in colorDict.keys(): #if pixel color is in dictionary key
                    colorDict[bgr].append( (y, x) )
                else:
                    colorDict[bgr] = [ (y, x) ]
                
        return len(colorDict.keys())
    
    def __kmeansColorCluster(self, image, clusters, rounds):
        """
        Parameters
            image <np.ndarray> : image
            clusters <int> : number of clustering
            rounds <int> : how many iterate kmeans clustering
        returns
            clustered Image <np.ndarray>
            SSE <float> : Sum of Squared Error
        """
        
        height, width = image.shape[:2]
        print("H, W:", height, width)
        samples = np.zeros([ height * width, 3 ], dtype=np.float32)
        
        count = 0
        for x in range(height):
            for y in range(width):
                samples[count] = image[x][y]
                count += 1
        
        '''
        # compactness : SSE
        # labels : array about label, show like 0, 1
        # centers : Cluster centroid Array
        '''
        compactness, labels, centers = cv2.kmeans(
                    samples, # 학습 데이터 정렬, data type = np.float32
                    clusters, # 군집 개수
                    None, # 각 샘플의 군집 번호 정렬
                    
                    
                    # criteria (종료 기준) : 3 element tuple (method, max_iter, epsilon)
                    
                    # method
                    # TERM_CRITERIA_EPS = 특정 정확도에 도달하면 알고리즘 반복 종료
                    # TERM_CRITERIA_MAX_ITER = 특정 반복 횟수 지나면 알고리즘 반복 종료
                    # 두 개 합 = 위 어느 조건이라도 만족하면 종료
                    
                    # max_iter = 최대 반복 횟수 지정
                    # epsilon = 요구되는 특정 정확도
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                                100000, # max_iter 
                                0.0001), # epsilon 
                    # attempts : 다른 initial centroid 이용해 반복 실행할 횟수
                    attempts = rounds, 
                    
                    # flags : To set the Initial Centroids
                    # cv2.KMEANS_RANDOM_CENTERS > 랜덤 선택
                    # cv2.KMEANS_PP_CENTERS > K-Means++ 알고리즘
                    # cv2.KMEANS_USE_INITIAL_LABELS > 사용자 선택
                    # 중 하나 선택
                    
                    flags = cv2.KMEANS_PP_CENTERS)
        
        centers = np.uint8(centers)
        self.np_clustered_colors = centers
        res = centers[labels.flatten()]
        
        return res.reshape((image.shape)), round( compactness ** 0.5 // 10, 2 )
    
    
    def __matchColors(self, colorImage, *matchColors):
        """
        Parameters
            colorImage <np.ndarray> : Image
            matchColors <np.ndarray in tuple> : matching color list (BGR type)
        returns
            img <np.ndarray> : Painted Image
        """
        
        # 여러 색상 리스트중에서 해당 색상과 가장 비슷한 색상을 반환
        def getSimilarColor(color, colors):
            """
            Parameters
                color <np.ndarray> : one color (BGR type)
                colors <np.ndarray> : matching color list
            returns
                similarColor <np.ndarray> : most similar color (BGR type)
            """
            
            # 수정 21.07.06
            min_dist = 255 * 255 * 255 * 1.0
            min_index = 0
            for idx, c in enumerate( colors ):
                dist, sum_c, sum_color = self.__colorDistance(c, color)
                if dist <= min_dist:
                    min_index = idx
                    min_dist = dist
            index = min_index
            
            return colors[ index ]
        
        img = colorImage.copy()
        
        clusteredColor, paintingColor = matchColors
        
        colorDict = {}
        imageColors = []
        for y, row in enumerate(colorImage):
            for x, color in enumerate(row):
                t_color = tuple(color)
                if t_color in colorDict:
                    img[y][x] = colorDict[t_color]
                    continue
                
                color = np.array( [int(x) for x in color] )
                
                similarColor = getSimilarColor(color, clusteredColor)
                
                # clustered color를 지정된 color와 매칭
                similarColor = getSimilarColor(similarColor, paintingColor)
                
                
                img[y][x] = similarColor
                colorDict[t_color] = similarColor
                imageColors.append( similarColor )
                
        def setClusteredColorName(colorList):
            colorName = []
            colorList = [ tuple([x for x in color]) for color in colorList ]
            # print(colorList)
            for rgb in set(colorList):
                hex = self.__bgr2hex( (rgb[2], rgb[1], rgb[0]) )
                idx = self.hexColorCode.index(hex)
                colorName.append( self.colorName[idx] )
            
            return colorName
        self.list_clustered_color_names = setClusteredColorName(imageColors)
        
        return img
    
    def __colorDistance(self, fst, snd):
        '''
        # https://dev.to/tejeshreddy/color-difference-between-2-colours-using-python-182b
        '''
        from colormath.color_objects import sRGBColor, LabColor
        from colormath.color_conversions import convert_color
        from colormath.color_diff import delta_e_cie2000

        # Red Color
        color1_rgb = sRGBColor( fst[2], fst[1], fst[0] )
        # Blue Color
        color2_rgb = sRGBColor( snd[2], snd[1], snd[0] )
        # Convert from RGB to Lab Color Space
        color1_lab = convert_color(color1_rgb, LabColor)
        # Convert from RGB to Lab Color Space
        color2_lab = convert_color(color2_rgb, LabColor)
        # Find the color difference
        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        # print("The difference between the 2 color = ", delta_e)
        return delta_e, sum(fst), sum(snd)
    
    # BGR Color tuple convert to Hex Color String Code
    def __bgr2hex(self, bgr):
        b, g, r = bgr
        return ('%02x%02x%02x' % (b, g, r)).upper()
        
    # Hex Color String Code convert to BGR Color np.array
    def __hex2bgr(self, hex):
        return np.array( [int(hex[i:i+2], 16) for i in (4, 2, 0)] ) 
        
def imageExpand(image, guessSize=False, size = 3):
    """
    Parameters
        image <np.ndarray> : image
        guessSize <boolean> : expand as appropriate size (default = False)
        size <int> : Size to expand (default = 3)
    returns
        image <np.ndarray> : expanded image
    """
    if guessSize : 
        length = max(image.shape[1], image.shape[0])
        size = ( 5000 // length ) + 1
    #       INTER_LANCZOS4
    image = cv2.resize(image, None, fx=size, fy=size, interpolation=cv2.INTER_LINEAR)
    
    return  image
    
        
if __name__ == "__main__":
    '''
    * How to Use?
    
    # 클래스 선언
    painting = Painting( "./imagePath/image.jpg")
    
    # Reduce Color and Blurring
    blurImage = painting.blurring(  div = 8, 
                                    radius = 10, 
                                    sigmaColor =20, 
                                    medianValue=7)
    
    # Color Clustering using K-Means
    clusteredImage = painting.colorClustering( blurImage, cluster = 16)
    
    # 이미지 확장, Way 1 or 2 < Select one
    expandedImage = imageExpand(clusteredImage, size = 4)
    # 확장된 이미지에서 변형된 색상을 군집화된 색상과 매칭
    similarMap = painting.expandImageColorMatch(expandedImage)
    # 군집화된 색상을 지정된 색상과 가장 비슷한 색상으로 매칭
    np_color_matched_img = painting.getPaintingColorMap(similarMap)
    
    # 이미지 색상 개수 확인
    number_of_color = painting.getNumberOfColor(np_color_matched_img)
    print("Number of Color :", number_of_color)
    '''
    pass
    
    