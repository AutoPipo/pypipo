# -*- coding: utf-8 -*-

import cv2
import math
import colorsys
import numpy as np
from datetime import datetime

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from .paint_color_rgb_code  import *


MAX_RGB_VALUE = 255.0
MAX_HUE = 360.0
MAX_SATURATION_LIGHTNESS = 100.0

# Constants for weights
HUE_WEIGHT = 0.5
SATURATION_WEIGHT = 0.3
LIGHTNESS_WEIGHT = 0.2

# Constants for normalization
HUE_NORMALIZATION = 180.0
SATURATION_NORMALIZATION = 100.0
LIGHTNESS_NORMALIZATION = 100.0

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


def rgb_to_lab(rgb):
    b, g, r = [x / 255.0 for x in rgb]
    r = _gamma_correction(r)
    g = _gamma_correction(g)
    b = _gamma_correction(b)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    x = _xyz_to_lab(x)
    y = _xyz_to_lab(y)
    z = _xyz_to_lab(z)

    l = max(0.0, 116.0 * y - 16.0)
    a = (x - y) * 500.0
    b = (y - z) * 200.0

    return l, a, b

def color_difference_cie76(color1, color2):
    l1, a1, b1 = color1
    l2, a2, b2 = color2

    delta_l = l2 - l1
    delta_a = a2 - a1
    delta_b = b2 - b1

    return math.sqrt(delta_l**2 + delta_a**2 + delta_b**2)

def _gamma_correction(value):
    if value <= 0.04045:
        return value / 12.92
    else:
        return ((value + 0.055) / 1.055) ** 2.4

def _xyz_to_lab(value):
    if value > 0.008856:
        return value ** (1.0 / 3.0)
    else:
        return (value * 903.3 + 16.0) / 116.0

def get_color_distance(color1, color2):
    '''
    Calculate the Euclidean distance between two colors in either RGB or HSL format.

    Parameters
    ----------
    color1 : tuple[float, float, float]
        RGB or HSL values of the first color.
    color2 : tuple[float, float, float]
        RGB or HSL values of the second color.

    Returns
    -------
    distance : float
        Euclidean distance between the two colors.
    '''
    v1, v2, v3 = color1
    p1, p2, p3 = color2
    distance = math.sqrt((v1 - p1)**2 + (v2 - p2)**2 + (v3 - p3)**2)
    return distance

def find_closest_color(target_color, color_list):
    '''
    Find the closest color to the target color in a list, considering either RGB or HSL format.

    Parameters
    ----------
    target_color : tuple[float, float, float]
        RGB or HSL values of the target color.
    color_list : list[tuple[float, float, float]]
        List of RGB or HSL values representing colors.

    Returns
    -------
    closest_color : tuple[float, float, float]
        RGB or HSL values of the color closest to the target color.
    '''
    min_distance = float('inf')
    closest_color = None

    for color in color_list:
        # distance = calculate_color_difference(target_color, color)
        # distance = color_difference_cie76(target_color, color)
        distance = delta_e_cie2000(target_color, color)
        if distance < min_distance:
            min_distance = distance
            closest_color = color

    return closest_color

def find_most_similar_paint_bgr_color(color_bgr):
    '''
    Find the most similar paint color in BGR format to the provided BGR color.

    Parameters
    ----------
    color_bgr : tuple[int, int, int]
        BGR values of the target color.

    Returns
    -------
    matched_color_bgr : tuple[int, int, int]
        BGR values of the most similar paint color.
    '''
    # Get a list of fixed paint colors in HSL format
    fixed_paint_hsl_list = [bgr_to_hsl(color) for color in get_fixed_painted_rgb_color_hex()]
    # Convert the target color to HSL format
    target_color_hsl = bgr_to_hsl(color_bgr)

    # testcode -- 
    # fixed_paint_lab_list = [rgb_to_lab(color) for color in get_fixed_painted_rgb_color_hex()]
    # color1_lab = rgb_to_lab(color_bgr)
    
    fixed_paint_lab_list = [bgr_to_lab((color[2], color[1], color[0])) for color in get_fixed_painted_rgb_color_hex()]
    color1_lab = bgr_to_lab(color_bgr)
    matched_color_lab = find_closest_color(color1_lab, fixed_paint_lab_list)
    matched_color_bgr = lab_to_bgr2(matched_color_lab)
    # end --

    # Find the closest color in HSL format
    #   matched_color_hsl = find_closest_color(target_color_hsl, fixed_paint_hsl_list)
    # Convert the closest color back to BGR format
    #   matched_color_bgr = hsl_to_bgr(matched_color_hsl)

    return matched_color_bgr

def calculate_color_difference(color1, color2):
    """
    Calculate the normalized color difference between two colors.

    Parameters
    ----------
    color1 : tuple[float, float, float]
        First color in HSL format (Hue, Saturation, Lightness).
    color2 : tuple[float, float, float]
        Second color in HSL format (Hue, Saturation, Lightness).

    Returns
    ----------
        float: Normalized color difference between 0 and 1.
    """

    # Unpack HSL components
    h1, s1, l1 = color1
    h2, s2, l2 = color2

    # Normalize the hue difference to a range between 0 and 1
    hue_diff = min(abs(h1 - h2), MAX_HUE - abs(h1 - h2)) / HUE_NORMALIZATION
    # Normalize the saturation difference to a range between 0 and 1
    sat_diff = abs(s1 - s2) / SATURATION_NORMALIZATION
    # Normalize the lightness difference to a range between 0 and 1
    light_diff = abs(l1 - l2) / LIGHTNESS_NORMALIZATION
    # Weighted sum of normalized differences with user-defined weights
    weighted_difference = (HUE_WEIGHT * hue_diff + 
                            SATURATION_NORMALIZATION * sat_diff + 
                            LIGHTNESS_WEIGHT * light_diff)

    return weighted_difference

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

def lab_to_bgr2(lab):
    # Convert LAB to RGB
    rgb = convert_color(lab, sRGBColor)

    # Extract RGB values in the range [0, 1]
    r = max(0, min(rgb.rgb_r, 1))
    g = max(0, min(rgb.rgb_g, 1))
    b = max(0, min(rgb.rgb_b, 1))

    # Scale to the range [0, 255]
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)

    return b, g, r

def lab_to_bgr(lab):
    l, a, b = lab

    y = (l + 16.0) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0

    x = _lab_to_xyz(x)
    y = _lab_to_xyz(y)
    z = _lab_to_xyz(z)

    r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252

    r = _xyz_to_rgb(r)
    g = _xyz_to_rgb(g)
    b = _xyz_to_rgb(b)

    return int(b * 255), int(g * 255), int(r * 255)

def _lab_to_xyz(value):
    if value > 0.2068966:
        return value**3
    else:
        return (value - 16.0 / 116.0) / 7.787

def _xyz_to_rgb(value):
    if value > 0.0031308:
        return 1.055 * (value**(1.0 / 2.4)) - 0.055
    else:
        return value * 12.92

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

def bgr_to_lab(bgr):
    rgb = (bgr[2], bgr[1], bgr[0])
    lab1 = convert_color(sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0), LabColor)
    return  lab1

def rgb_to_ciede2000(rgb1, rgb2):
    # Convert RGB to LAB
    lab1 = convert_color(sRGBColor(rgb1[0]/255.0, rgb1[1]/255.0, rgb1[2]/255.0), LabColor)
    lab2 = convert_color(sRGBColor(rgb2[0]/255.0, rgb2[1]/255.0, rgb2[2]/255.0), LabColor)

    # Calculate CIEDE2000 color difference
    delta_e = delta_e_cie2000(lab1, lab2)

    return delta_e