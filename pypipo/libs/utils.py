# -*- coding: utf-8 -*-

import cv2
import numpy as np

# BGR Color tuple convert to Hex Color String Code
def bgr_to_hex(bgr):
    b, g, r = bgr
    return ('%02x%02x%02x' % (b, g, r)).upper()
    
# Hex Color String Code convert to BGR Color np.array
def hex_to_bgr(hex):
    return np.array([int(hex[i:i + 2], 16) for i in (4, 2, 0)])

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


def img_save(save_path, save_image):
    """Save output image
    Parameters
    ----------
    save_path : str
        File path that want to save image
    save_image : np.ndarray
        Image object
    """
    cv2.imwrite(save_path, save_image)
    return 


def check_range(value, minimum, maximum, name="input"):
    """
    Check if the given value is within the specified range and raise an exception if it's not.

    Parameters
    ----------
    value : int or float 
        The value to be checked.
    minimum : int or float
        The allowed minimum value.
    maximum : int or float
        The allowed maximum value.
    name : string, Optional (default="input")
        The string representing name of value in ValueError message.
    
    Raises
    ----------
    ValueError
        Exception raised if the input value is not within the specified range.

    Returns
    ----------
    Void
    """
    if value < minimum or value > maximum:
        raise ValueError(f"The {name} value {value} is outside the allowed range ({minimum} ~ {maximum}).")
    

def nearest_odd_integer(value):
    """
    Returns the nearest odd integer to the given value.

    Parameters
    ----------
    value : float or int
        The input value for which the nearest odd integer should be found.

    Returns
    ----------
    return : int 
        The nearest odd integer to the given value. 
        If the input value is itself an odd integer, it will be returned.
    """
    nearest_even = round(value)
    return nearest_even + 1 if nearest_even % 2 == 0 else nearest_even


def division_filter(image, divisor):
    ''' simplify image pixels with division
    Parameters
    ----------
        image : np.ndarray
            Input image
        divisor : int (0 <= divisor <= 255)
            A value for dividing color pixel values

    Returns
    ----------
        image : np.ndarry
            Blurred image
    '''
    check_range(divisor, 0, 255, name="div")
    return image // divisor * divisor + divisor // 2