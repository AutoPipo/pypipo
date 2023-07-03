# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
from tqdm import trange
from collections import defaultdict
from scipy.spatial import distance as dist


class Painting:
    """Change image to painting image.
    
    Parameters
    ----------
    filepath : str
        File path that you want to convert

    Attributes
    ----------
    original_img : np.array
        Input original image
    painting : np.array
        Color clustered image (Applied K-Means Algorithm)
    color_rbg_values : np.array
        Clustered color data list
    """

    def __init__(self, filepath):
        self.original_img = cv2.imread(filepath)
        self.clustered_colors = np.array([])
        return 
    
    def run(self, 
            number = 16, 
            attempts = 1,
            is_upscale = False,
            target_size = 3,
            div = 8, 
            sigma = 20):
        """Cluster image color with k-means algorithm.

        Parameters
        ----------
        number : int, optional (default: 16)
            Number of color clustered
        attempts : int, optional (default: 1)
            How many iterate try to k-means clustering
        is_upscale : bool, optional (default: False)
            Expand size of image
        target_size : int, optional (default: 3)
            Size that want to expand image.
            If you want to guess the proper size, set size value under 1.
        div : int, optional (default: 8)
            Reducing numbers of color on image
        sigma : int, optional (default: 20)
            bilateralFilter Parameter

        Returns
        ----------
        painting : np.ndarray
            Color clustered image
        color_index_map : np.ndarray
            a Array that contains clustered color indexs.
        """

        target_image = self.__blurring(div, sigma)

        if is_upscale:
            target_image = self.__expand_image(target_image, target_size = target_size)
        
        painting, color_index_map = self.__cluster_color_with_kmeans(target_image, 
                                                                    number_of_color = number, 
                                                                    attempts = attempts)
        
        return painting, color_index_map
    
    def __blurring(self, div, sigma):
        """Image blurring

        Parameters
        ----------
        div : int
            Reducing numbers of color on image
        sigma : int
            bilateralFilter Parameter

        Returns
        -------
        blurred_image : np.ndarray
            blurred Image
        """

        BILATERAL_FILTER_RADIUS = -1  # Auto decision by sigmaSpace
        BILATERAL_FILTER_SIGMACOLOR_MIN = 10
        BILATERAL_FILTER_SIGMACOLOR_MAX = 120
        
        qimg = self.original_img.copy()  # copy original image
        
        sigma = max(sigma, BILATERAL_FILTER_SIGMACOLOR_MIN)
        sigma = min(sigma, BILATERAL_FILTER_SIGMACOLOR_MAX)
        
        # bilateral blurring
        blurred_image = cv2.bilateralFilter(qimg, BILATERAL_FILTER_RADIUS, sigma, sigma)
        
        # reduce numbers of color
        blurred_image = blurred_image // div * div + div // 2
        return blurred_image
    
    def __cluster_color_with_kmeans(self, image, number_of_color, attempts):
        """Cluster image color with k-means algorithm.

        Parameters
        ----------
        image : np.ndarray
            Input image
        number_of_color : int
            Number of color clustered
        attempts : int
            How many iterate try to k-means clustering

        Returns
        ----------
        color_clustered_image : np.ndarray
            Color clustered image
        color_index_map : np.ndarray
            a Array that contains clustered color indexs.
        """
        height, width = image.shape[:2]
        
        # transform color data to use k-means algorithm
        # need to trnasform into two-dimensional array
        # [[B, G, R], [B, G, R] ... , [B, G, R]]
        training_data_samples = image.reshape((height * width, 3)).astype(np.float32)

        # sse : Sum of squared error
        # labels : Array about label, show like 0, 1
        # centers : Cluster centroid array
        sse, labels, centers = cv2.kmeans(
                                        training_data_samples,  # Align training data, data type = np.float32
                                        number_of_color,        # number of cluster
                                        None,                   # Sort the cluster numbers for each sample
                                        
                                        # TERM_CRITERIA_EPS : End iteration when a certain accuracy is reached
                                        # TERM_CRITERIA_MAX_ITER : End iteration after a certain number of iterations
                                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                                                    100000, # max_iter : Max number of iterations
                                                    0.0001), # epsilon : Specific Accuracy Required
                                        attempts = attempts,  # Number of iterations to run using different initial centroids
                                        
                                        # flags : To set the Initial Centroids
                                        # cv2.KMEANS_RANDOM_CENTERS : Random
                                        # cv2.KMEANS_PP_CENTERS : K-Means++ Algorithm
                                        # cv2.KMEANS_USE_INITIAL_LABELS : User selection
                                        flags = cv2.KMEANS_PP_CENTERS)
        
        # a Array that contains clustered color indexs.
        # it has same shape as Oringinal image, but the value it contains is a single-dimension.
        # it will be used to draw line along the colors.
        color_index_map = labels.reshape((height, width))

        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        self.clustered_colors = centers

        # for returns
        sse = round(sse ** 0.5 // 10, 2)
        color_clustered_image = res.reshape((image.shape))
        return color_clustered_image, color_index_map
   
    def __expand_image(self, image, target_size):
        """Expand image size

        Parameters
        ----------
        image : np.ndarray
            Input image
        target_size : int
            Size that want to expand image.
            If you want to guess the proper size, set size value under 1.

        Returns
        ----------
        output : np.ndarray
            Expanded image
        """
        STANDARD_SIZE_OF_IMAGE = 5000

        if target_size < 1:
            max_image_length = max(image.shape[1], image.shape[0])
            target_size = (STANDARD_SIZE_OF_IMAGE // max_image_length) + 1

        output = cv2.resize(image, None, fx = target_size, fy = target_size, interpolation = cv2.INTER_LINEAR)
        return output
    
    def get_clustered_color_info(self, painting_img):
        '''Extract color at image.

        Parameters
        ----------
        painting_img : np.ndarray
            Color clustered image

        Returns
        ----------
        color_indexs : list[str]
            Color index number list
        color_rbg_values : np.ndarray
            Clustered color data list
        '''
        
        def create_color_location_dict(image):
            # Color list return function
            color_location_dict = defaultdict(list)  # key: BGR color, value: (x, y) location at image
            for y, row in enumerate(image):
                for x, bgr in enumerate(row):
                    color_location_dict[tuple(bgr)].append((x, y))
            return color_location_dict

        color_indexs, color_rbg_values = [], []
        for idx, color in enumerate(list(create_color_location_dict(painting_img).keys()),  1):
            color_indexs.append(str(idx))
            color_rbg_values.append(color)

        return color_indexs, color_rbg_values
    

class LineDrawing:
    """Draw line on image with color boundary
    
    Parameters
    ----------
    color_index_map : np.ndarray
        a Array that contains clustered color indexs.
        it might be returned from Painting Process.

    Attributes
    ----------
    web : np.array
        Remain only lines from image color boundary, white background
    """
    def __init__(self, color_index_map):
        self.color_index_map = color_index_map
        self.WHITE_COLOR = 255
        self.BLACK_COLOR = 0
    
    def run(self, outline = True):
        """Draw line on image

        Parameters
        ----------
        outline : bool, optional (default: True)
            Select that want to draw outline on web image

        Returns
        ----------
        self.web : np.ndarray
            Gray scale that line drawn on white background image
        """
        web = self.__draw_line()

        if outline:
            web = self.__draw_outline(web)
            
        return web
    
    def __draw_line(self):
        """Draw line with color boundary from painting image

        Parameters
        ----------
        painting : np.ndarray
            Input painting image

        Returns
        ----------
        web : np.ndarray
            Gray scale image that line drawn
        """

        # Find color index difference by comparing row and columns.
        hor_diff = self.__get_diff(self.color_index_map)
        ver_diff = self.__get_diff(self.color_index_map.T)

        # rotate 90 degree to fit shape with hor_diff
        ver_diff = ver_diff.T

        # merge horizontal and vertical difference by element-wise operation
        diff = ver_diff + hor_diff
        
        # set pixel color to black if there is a difference
        web = np.where(diff != 0, self.BLACK_COLOR, self.WHITE_COLOR)
        return web
    
    def __draw_outline(self, web):
        """Draw outline on image
        """
        web[0:2], web[-3:-1], web[:,0:2], web[:,-3:-1] = 0, 0, 0, 0
        return web

    def __get_diff(self, map):
        """Draw outline on image
        Parameters
        ----------
        map : np.ndarray
            a array that contains simplified color index
        
        Returns
        ---------
        diff : np.ndarray
            a array that contains each row value diffences.
        """

        diff = np.zeros(map.shape) + self.WHITE_COLOR

        # subtracts current row and next row
        for y, row in enumerate(map[:-1]):
            next_row = map[y + 1]
            diff[y] = row - next_row

        return diff
    
    def get_image_lab(self, color_rbg_values, painting_img):
        '''Get inner circle radius, center coordinates

        Parameters
        ----------
        color_rbg_values : np.ndarray
            Clustered color data list
        painting_img : np.ndarray
            Color clustered image

        Returns
        ----------
        lab_image : np.ndarray
            @@@
        lab : np.ndarray
            @@@
        '''
        lab = np.zeros((len(color_rbg_values), 1, 3), dtype="uint8")
        for i in range(len(color_rbg_values)):
            lab[i] = color_rbg_values[i]

        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)

        # Use LAB as color space for color detection
        img_lab = cv2.cvtColor(painting_img, cv2.COLOR_BGR2LAB)

        return img_lab, lab
    

class ColorspaceIndexing:
    """Color indexing at colorspace
    
    Parameters
    ----------
    painting_img : np.ndarray
        Color clustered image
    web_img : np.ndarray
        Line drawn edge of color at color clustered image
    color_indexs : list[str]
        Color index number list
    color_rbg_values : list[tuple]
        RGB values
    """
    def __init__(self, painting_img, web_img, color_indexs, color_rbg_values):
        self.painting_img = painting_img
        self.web_img = web_img
        self.color_indexs = color_indexs
        self.color_rbg_values = color_rbg_values
        self.WHITE_COLOR = 255

        self.NUMBERING_MIN_AREA = 80   # numbering minimum area
        self.NUMBERING_MIN_RADIUS = 8  # numbering minimum radius
        return 
    
    def __get_contours_information_from_web(self, web_image):
        """Get contours, hierarchy, image_bin from web image

        Parameters
        ----------
        web_image : np.ndarray
            Line drawn image
        """
        web_image = web_image.astype(np.uint8)  # have to convert grayscale
        _, image_bin = cv2.threshold(web_image, 127,255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours, hierarchy, image_bin
    
    def set_opacity_base_image(self, base_image, wrap_image, opacity = 0.2):
        '''Apply opacity base image, and put under the wrap image.

        Parameters
        ----------
        base_image : np.ndarray
            Base image
        wrap_image : np.ndarray
            Wrap image on base image
        opacity : float, optional (default: 0.2)
            Opacity value for base image

        Returns
        ----------
        output : np.ndarray
            Base image which is applied opacity and put under the wrap image.
        '''
        output = cv2.addWeighted(base_image, opacity, wrap_image, (1 - opacity), 0, dtype = cv2.CV_32F)
        return output
    
    def __get_circle_radius_center(self, raw_dist):
        '''Get inner circle radius, center coordinates

        Parameters
        ----------
        raw_dist : np.ndarray
            @@@

        Returns
        ----------
        radius : float
            Radius of inner circle
        center : tuple
            Center coordinate of inner circle
        '''
        dist_transform, _ = cv2.distanceTransformWithLabels(raw_dist, cv2.DIST_L2, maskSize=5)
        _, radius, _, center = cv2.minMaxLoc(dist_transform)
        return radius, center
    
    def check_avg_color_inside_colorspace(self, lab_image, contour, lab):
        '''Check which color, through mean of color value in contour.

        Parameters
        ----------
        lab_image : np.ndarray
            @@@
        contour : np.ndarray
            @@@
        lab : np.ndarray
            @@@

        Returns
        ----------
        color_index : str
            Color index string
        '''
        mask = np.zeros(lab_image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(lab_image, mask=mask)[:3]
        min_dist = (np.inf, None)

        for i, row in enumerate(lab):
            distance = dist.euclidean(row[0], mean)
            if distance < min_dist[0]:
                min_dist = (distance, i)
        color_index = self.color_indexs[min_dist[1]]
        return color_index
    
    def __set_label_inside_colorspace(self, image, num, center_point, radius):
        '''Put color index string in inner circle center point.
           https://github.com/bsdnoobz/opencv-code/blob/master/shape-detect.cpp

        Parameters
        ----------
        image : np.ndarray
            Background image
        num : str
            Color index string
        center_point : tuple
            Coordinate of inner circle center
        radius : float
            Radius of inner circle

        Returns
        ----------
        None
        '''
        fontface = cv2.FONT_HERSHEY_SIMPLEX

        # TODO: Create constants as variables
        scale = 0.5 if radius / 100 < 0.5 else radius / 100 # 0.6
        thickness = 1 if scale < 0.8 else 2

        textsize = cv2.getTextSize(num, fontface, scale, thickness)[0]
        center_point = (int(center_point[0]-(textsize[0]/2)+1), int(center_point[1]+(textsize[1]/2)))

        cv2.putText(image, num, center_point, fontface, scale, (150, 150, 150), thickness, 8)
        return 
    
    def __numbering_colorspace_from_contours(self, 
                                             background_img, 
                                             image_bin, 
                                             contours, 
                                             hierarchy, 
                                             img_lab, 
                                             lab):
        '''looping contours list, and put color index label in each contour.

        Parameters
        ----------
        background_img : np.ndarray
            Background image
        image_bin : np.ndarray
            @@@
        contours : tuple
            @@@
        hierarchy : np.ndarray
            @@@
        img_lab : np.ndarray
            @@@
        lab : np.ndarray
            @@@

        Returns
        ----------
        background_img : np.ndarray
            Image that filled color index string each contours.
        '''
        # TODO: more faster
        for idx in trange(len(contours), file=sys.stdout, desc='Numbering Process'):
            contour = contours[idx]

            # Ignore areas below a certain size
            if cv2.contourArea(contour) < self.NUMBERING_MIN_AREA:
                continue

            chlidren = [i for i, hierarchy_obj in enumerate(hierarchy[0]) if hierarchy_obj[3] == idx]

            raw_dist = np.zeros(image_bin.shape, dtype=np.uint8)
            cv2.drawContours(raw_dist, contour, -1, (255, 255, 255), 1)
            cv2.fillPoly(raw_dist, pts =[contour], color=(255, 255, 255))
            cv2.fillPoly(raw_dist, pts =[contours[i] for i in chlidren], color=(0, 0, 0))

            radius, center = self.__get_circle_radius_center(raw_dist)

            # Ignore radius below a certain length
            if radius < self.NUMBERING_MIN_RADIUS:
                continue

            if center is not None:
                cv2.drawContours(background_img, [contour], -1, (150, 150, 150), 1)

                # 내접원 확인용(주석 풀면 활성화)
                # cv2.circle(img, center, int(radius), (0, 255, 0), 1, cv2.LINE_8, 0)

                # Show the color detected inside the contour
                color_text = self.check_avg_color_inside_colorspace(img_lab, contour, lab)

                center_point = (center[0], center[1])
                self.__set_label_inside_colorspace(background_img, color_text, center_point, radius)
                
        return background_img
    
    # TODO: size fix by image size
    def __put_color_label_lefttop_side(self, background_img):
        '''Put color label at left top of image.

        Parameters
        ----------
        background_img : np.ndarray
            Background image

        Returns
        ----------
        background_img : np.ndarray
            Image that filled color index label each color.
        '''
        # put color label, at left top side on image
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        LINE_SCALE = 1  # 0.6
        LINE_THICKNESS = 2  # 2

        for idx in range(len(self.color_rbg_values)):
            cv2.putText(background_img, self.color_indexs[idx], (20, 40*(idx+1)), 
                        fontface, LINE_SCALE, (50, 50, 50), LINE_THICKNESS, 8)
            cv2.rectangle(background_img, (60, 40*(idx+1)-20), (90, 40*(idx+1)), 
                          tuple([int(i) for i in self.color_rbg_values[idx]]), -1, 8)
            
        return background_img
    
    def run(self, img_lab, lab, color_label = True):
        '''Main process of this class.

        Parameters
        ----------
        img_lab : np.ndarray
            @@@
        lab : np.ndarray
            @@@
        color_label : bool
            Whether to display a label

        Returns
        ----------
        output : np.ndarray
            Result image of process
        '''
        contours, hierarchy, image_bin = self.__get_contours_information_from_web(self.web_img.copy())
        # Make the output image white
        background_img = np.zeros(self.painting_img.copy().shape) + self.WHITE_COLOR
        background_img = self.set_opacity_base_image(self.painting_img, background_img)

        # Rendering output
        output = self.__numbering_colorspace_from_contours(background_img, image_bin, contours, hierarchy, img_lab, lab)
        if color_label:
            output = self.__put_color_label_lefttop_side(output)

        return output
    