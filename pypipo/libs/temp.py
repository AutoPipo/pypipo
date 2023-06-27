# ���⿡ Ȯ���� �̹����� Ŭ������ Į�� ��Ī 
def expandImageColorMatch(expand_image, colors_list):
    np_color_clustered_img =  __matchColors(expand_image, colors_list)
    return np_color_clustered_img

def __matchColors(colorImage, *matchColors):
    """
    Parameters
        colorImage <np.ndarray> : Image
        matchColors <np.ndarray in tuple> : matching color list (BGR type)
    returns
        img <np.ndarray> : Painted Image
    """
    
    # ���� ���� ����Ʈ�߿��� �ش� ����� ���� ����� ������ ��ȯ
    def getSimilarColor(color, colors):
        """
        Parameters
            color <np.ndarray> : one color (BGR type)
            colors <np.ndarray> : matching color list
        returns
            similarColor <np.ndarray> : most similar color (BGR type)
        """
        
        # ���� 21.07.06
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
            
            # clustered color�� ������ color�� ��Ī
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