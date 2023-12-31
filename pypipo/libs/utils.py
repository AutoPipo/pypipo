# -*- coding: utf-8 -*-

import cv2
import math
import colorsys
import numpy as np
from datetime import datetime

from ._colormath.color_objects import *
from ._colormath.color_conversions import convert_color
from ._colormath.color_diff import delta_e_cie2000
from .paint_color_rgb_code  import *

MAX_RGB_VALUE = 255
MAX_HUE = 360
MAX_SATURATION_LIGHTNESS = 100

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

def bgr_to_hsl(bgr):
    '''
    Convert BGR to HSL.

    Parameters
    ----------
    bgr : tuple[int, int, int]
        BGR value

    Returns
    -------
    hsl : tuple[float, float, float]
        HSL value
    '''
    # Normalize BGR values to the range between 0 and 1
    b, g, r = [x / MAX_RGB_VALUE for x in bgr]
    # Convert BGR to HSL using the colorsys library
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Convert h, l, s to the specified format
    h *= MAX_HUE
    l *= MAX_SATURATION_LIGHTNESS
    s *= MAX_SATURATION_LIGHTNESS
    
    return h, s, l

def hsl_to_bgr(hsl):
    '''
    Convert HSL to BGR.

    Parameters
    ----------
    hsl : tuple[float, float, float]
        HSL values (Hue, Saturation, Lightness)

    Returns
    -------
    bgr : tuple[int, int, int]
        BGR values (Blue, Green, Red)
    '''
    # Unpack HSL values
    h, s, l = hsl

    # Normalize H, S, L values to the range between 0 and 1
    h_normalized = h / MAX_HUE
    s_normalized = s / MAX_SATURATION_LIGHTNESS
    l_normalized = l / MAX_SATURATION_LIGHTNESS

    # Convert HSL to RGB using the colorsys library
    r, g, b = colorsys.hls_to_rgb(h_normalized, l_normalized, s_normalized)

    # Convert RGB values to the range between 0 and 255
    r_int = int(r * MAX_RGB_VALUE)
    g_int = int(g * MAX_RGB_VALUE)
    b_int = int(b * MAX_RGB_VALUE)

    return b_int, g_int, r_int

def find_closest_color(target_color, color_list):
    '''
    Find the closest color to the target color in a list, with CIE2000.

    Parameters
    ----------
    target_color : tuple[float, float, float]
        RGB values of the target color.
    color_list : list[tuple[float, float, float]]
        List of RGB values representing colors.

    Returns
    -------
    closest_color : tuple[float, float, float]
        RGB values of the color closest to the target color.
    '''
    min_distance = float('inf')
    closest_color = None

    for color in color_list:
        distance = delta_e_cie2000(target_color, color)
        if distance < min_distance:
            min_distance = distance
            closest_color = color

    return closest_color

def lab_to_bgr(lab_color):
    '''
    Lab Object convert to BGR value.

    Parameters
    ----------
    lab_color : LAB Object

    Returns
    -------
        tuple[int, int, int]
            BGR values
    '''
    # Convert LAB to BGR
    rgb_color = convert_color(lab_color, sRGBColor)

    # Ensure RGB values are within the valid range [0, 1]
    r = _clamp_rgb_value(rgb_color.rgb_r)
    g = _clamp_rgb_value(rgb_color.rgb_g)
    b = _clamp_rgb_value(rgb_color.rgb_b)

    # Scale to the range [0, 255]
    r, g, b = _scale_to_255(r, g, b)

    return b, g, r

def _clamp_rgb_value(value):
    # Ensure RGB value is within the valid range [0, 1]
    return max(0, min(value, 1))

def _scale_to_255(*values):
    # Scale values to the range [0, 255]
    return tuple(int(v * MAX_RGB_VALUE) for v in values)

def bgr_to_lab(bgr):
    '''
    BGR value convert to Lab Object.
    '''
    lab = convert_color(sRGBColor(bgr[2]/MAX_RGB_VALUE, 
                                bgr[1]/MAX_RGB_VALUE, 
                                bgr[0]/MAX_RGB_VALUE), 
                        LabColor)
    return  lab

def find_most_similar_paint_bgr_color(target_bgr_color):
    '''
    Find the most similar paint color to the given BGR color.

    Parameters
    ----------
    target_bgr_color : tuple[int, int, int]
        BGR values of the target color.

    Returns
    -------
    most_similar_bgr_color : tuple[int, int, int]
        BGR values of the most similar paint color.
    '''

    # Convert the list of paint colors to Lab format
    fixed_paint_color_lab_list = [bgr_to_lab(tuple(reversed(color))) for color in get_fixed_painted_rgb_color_hex()]
    # Convert the target color to Lab format
    target_lab_color = bgr_to_lab(target_bgr_color)
    # Find the closest paint color
    closest_color_lab = find_closest_color(target_lab_color, fixed_paint_color_lab_list)
    # Convert the found color in Lab format to BGR
    most_similar_bgr_color = lab_to_bgr(closest_color_lab)
    
    return most_similar_bgr_color

def log_to_file(message, level='info', file_path='log.txt'):
    """
    Write logs to a file.

    Parameters
    ----------
    message : str
        Log message to be written.
    level : str, optional
        Log level (default : 'info')
    file_path : str, optional
        File path for writing logs (default : 'log.txt')

    Raises
    ------
    ValueError
        If an invalid log level is provided.

    Notes
    -----
    This function appends logs to the specified file, formatted with a timestamp and log level.

    Returns
    -------
    None
    """
    levels = ['info', 'warning', 'error']

    if level not in levels:
        raise ValueError("Invalid log level. Use one of: {}".format(', '.join(levels)))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = "[{}] [{}] {}".format(timestamp, level.upper(), message)

    with open(file_path, 'a') as file:
        file.write(log_message + '\n')

    return

