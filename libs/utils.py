
import cv2
import numpy as np

# 이미지 투명하게
def set_alpha_on_background_image(base_image, wrap_image, alpha = 0.2):
    '''
    Parameters
    ----------
    base_image : np.ndarray
        Base image
    wrap_image : np.ndarray
        Wrap image on base image
    alpha : float, optional (default: 0.2)
        Alpha value apply to base image

    Returns
    ----------
    output : np.ndarray
        Base image applied alpha + wrap_image
    '''
    output = cv2.addWeighted(base_image, alpha, wrap_image, (1 - alpha), 0, dtype = cv2.CV_32F)
    return output



# BGR Color tuple convert to Hex Color String Code
def bgr2hex(bgr):
    b, g, r = bgr
    return ('%02x%02x%02x' % (b, g, r)).upper()
    
# Hex Color String Code convert to BGR Color np.array
def hex2bgr(hex):
    return np.array( [int(hex[i:i+2], 16) for i in (4, 2, 0)] ) 



# counting numbers of color
def get_number_of_image_color(image):
    """Get number of image colors
    Parameters
    ----------
        image : np.ndarray
            Input image

    Returns
    ----------
        number_of_colors : int
            Number of colors
    """
    colorDict = {} # Key : Color Code / Values : Pixel Position
    for y, row in enumerate(image):
        for x, bgr in enumerate(row):
            bgr = tuple(bgr) # np.ndarray convert to tuple
            if colorDict == {}: # if dictionary in empty
                colorDict[ bgr ] = [(y, x)]
                continue
            
            if bgr in colorDict.keys(): #if pixel color is in dictionary key
                colorDict[bgr].append((y, x))
            else:
                colorDict[bgr] = [(y, x)]

    number_of_colors = len(colorDict.keys())  
    return number_of_colors