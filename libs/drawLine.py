'''
# Draw Line on Image based on Color Boundaries

# Start : 21.04.01
# Update : 21.06.12
# Author : Minku Koo
'''

import cv2
import numpy as np

class DrawLine:
    def __init__(self, image):
        '''
        Parameters
            image <np.ndarray> : Image Object
        '''
        self.colorImage = image
        self.lineMap = np.zeros(image.shape) + 255
    
    def getDrawLine(self):
        '''
        Line on White Image
        '''
        return self.__drawLine()
    
    def getLineOnImage(self):
        '''
        Line on Real Image
        '''
        return self.__lineOnImage()
    
    def __drawLine(self):
        for y, orgRow in enumerate(self.colorImage[:-1]):
            nextRow = self.colorImage[y+1]
            # 다음 row와 비교했을 때, 색상이 다른 index 추출
            compareRow = np.array( np.where((orgRow == nextRow) == False))
            for x in np.unique(compareRow[0]):
                # Convert to Black
                self.lineMap[y][x] = np.array([0, 0, 0])
                            
        width = self.lineMap.shape[1] # get Image Width
        for x in range(width-1):
            # 다음 column과 비교했을 때, 색상이 다른 index 추출
            compareCol = np.array( np.where((self.colorImage[:,x] == self.colorImage[:,x+1]) == False))
            for y in np.unique(compareCol[0]):
                # Convert to Black
                self.lineMap[y][x] = np.array([0, 0, 0])
        
        # threshold를 이용하여, 2차원 Image로 변환
        _, self.lineMap = cv2.threshold(self.lineMap, 199, 255, cv2.THRESH_BINARY)
        
        return self.lineMap
    
    
    def __lineOnImage(self):
        new_map = self.colorImage.copy()
        lineMap = self.lineMap // 255
        return np.multiply(new_map, lineMap) 
    
    def drawOutline(self, image):
        # 이미지 가장자리에 임의의 선을 그어줌
        # image[0], image[-1], image[:,0], image[:,-1] = 0, 0, 0, 0
        image[0:2], image[-3:-1], image[:,0:2], image[:,-3:-1] = 0, 0, 0, 0
        return image
    

def leaveOnePixel(lineImage):
    image = lineImage.copy()
    
    _, image = cv2.threshold(image, 200, 1, cv2.THRESH_BINARY_INV)
    skeleton = skeletonize(image)
    skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
    
    canvas = np.zeros(skeleton.shape) + 1
    
    return 255 - np.multiply( canvas, skeleton )
    
    
if __name__ == "__main__":
    '''
    * How to Use?
    
    drawLine = DrawLine(image)
    lineMap = drawLine.getDrawLine()
    outlines = drawLine.drawOutline(lineMap)
    lineOnImage = drawLine.getLineOnImage()
    '''
    pass
    