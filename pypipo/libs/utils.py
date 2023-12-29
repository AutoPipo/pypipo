# -*- coding: utf-8 -*-

import cv2
import math
import colorsys
import numpy as np
from datetime import datetime
from .paint_color_rgb_code  import *

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

def img_read(file_path):
    np_read_image = cv2.imread(file_path)
    return np_read_image

def img_save(save_path, save_image):
    cv2.imwrite(save_path, save_image)
    return 


def check_parameter_range(value, minimum, maximum, name="input"):
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
    None
    """
    if value < minimum or value > maximum:
        raise ValueError(f"The {name} value {value} is outside the allowed range ({minimum} ~ {maximum}).")
    
    return 
    

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
    nearest_odd = (nearest_even + 1) if (nearest_even % 2 == 0) else nearest_even
    return nearest_odd


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
        blurred_image : np.ndarry
            Blurred image
    '''
    check_parameter_range(divisor, 1, 255, name="div")
    blurred_image =  (image // divisor * divisor) + (divisor // 2)
    return blurred_image

def set_opacity_base_image(base_image, wrap_image, opacity = 0.3):
    '''Apply opacity base image, and put under the wrap image.

    Parameters
    ----------
    base_image : np.ndarray
        Base image
    wrap_image : np.ndarray
        Wrap image on base image
    opacity : float, optional (default: 0.3)
        Opacity value for base image

    Returns
    ----------
    output : np.ndarray
        Base image which is applied opacity and put under the wrap image.
    '''
    output = cv2.addWeighted(base_image, opacity, wrap_image, (1 - opacity), 0, dtype = cv2.CV_32F)
    return output

def rgb_to_hsl(bgr):
    RGB_TO_HSL_DIVIDED = 255.0
    b, g, r = [(x / RGB_TO_HSL_DIVIDED) for x in bgr]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s * 100, l * 100

def hsl_to_rgb(hsl):
    h, s, l = hsl
    h /= 360.0  # Hue를 0에서 1 사이의 값으로 정규화
    s /= 100.0  # Saturation을 0에서 1 사이의 값으로 정규화
    l /= 100.0  # Lightness를 0에서 1 사이의 값으로 정규화
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(b * 255), int(g * 255), int(r * 255)

def get_color_distance(color1, color2):
    v1, v2, v3 = color1
    p1, p2, p3 = color2
    distance = math.sqrt((v1 - p1)**2 + (v2 - p2)**2 + (v3 - p3)**2)
    return distance

def find_closest_color(target_color, color_list):
    min_distance = float('inf')
    closest_color = None

    for color in color_list:
        distance = get_color_distance(target_color, color)
        if distance < min_distance:
            min_distance = distance
            closest_color = color

    return closest_color

def find_most_similar_paint_rgb_color(color_rgb):
    # fixed_paint_rgb_list = get_fixed_painted_rgb_color_hex()
    fixed_paint_hsl_list = [rgb_to_hsl(color) for color in get_fixed_painted_rgb_color_hex()]
    target_color_hsl = rgb_to_hsl(color_rgb)
    matched_color = find_closest_color(target_color_hsl, fixed_paint_hsl_list)
    matched_color = hsl_to_rgb(matched_color)
    return matched_color

def log_to_file(message, level='info', file_path='log.txt'):
    """
    파일에 로그를 작성하는 함수

    Parameters:
    - message (str): 작성할 로그 메시지
    - level (str): 로그 레벨 (기본값: 'info')
    - file_path (str): 로그를 작성할 파일 경로 (기본값: 'log.txt')
    """
    levels = ['info', 'warning', 'error']

    if level not in levels:
        raise ValueError("Invalid log level. Use one of: {}".format(', '.join(levels)))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = "[{}] [{}] {}".format(timestamp, level.upper(), message)

    with open(file_path, 'a') as file:
        file.write(log_message + '\n')

