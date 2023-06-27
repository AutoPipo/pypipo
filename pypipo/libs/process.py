# -*- coding: utf-8 -*-

import cv2
import numpy as np

class Painting:
    """Change image to painting image.
    
    Parameters
    ----------
    img : np.ndarray
        Image that want to work

    Attributes
    ----------
    original_img : np.array
        Input original image
    painting : np.array
        Color clustered image (Applied K-Means Algorithm)
    colors : np.array
        Clustered color data list
    """

    def __init__(self, img):
        self.original_img = img
        self.painting = np.array([])
        self.colors = np.array([])
        return 
    
    def run(self, 
            # kmeans clustering
            k = 16, 
            attempts = 1,
            # expand size
            is_upscale = False,
            size = 3,
            # blurring
            blurring = False,
            div = 8, 
            sigma_color = 20):
        """Cluster image color with k-means algorithm.

        Parameters
        ----------
        k : int, optional (default: 16)
            Number of color clustered
        attempts : int, optional (default: 1)
            How many iterate try to k-means clustering
        is_upscale : bool, optional (default: False)
            Expand size of image
        size : int, optional (default: 3)
            Size that want to expand image.
            If you want to guess the proper size, set size value under 1.
        blurring : bool, optional (default: False)
            Blurring image
        div : int, optional (default: 8)
            Reducing numbers of color on image
        sigma_color : int, optional (default: 20)
            bilateralFilter Parameter

        Returns
        ----------
        self.painting : np.ndarray
            Color clustered image
        color_index_map : np.ndarray
            a Array that contains clustered color indexs.
        """

        if blurring:
            target_image = self.__blurring(div, sigma_color)
        else:
            target_image = self.original_img.copy()

        if is_upscale:
            target_image = self.__expand_image(target_image, size = size)
        
        self.painting, color_index_map = self.__cluster_color_with_kmeans(target_image, 
                                                            number_of_color = k, 
                                                            attempts = attempts)
        return self.painting, color_index_map
    
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
        self.colors = centers

        # for returns
        sse = round(sse ** 0.5 // 10, 2)
        color_clustered_image = res.reshape((image.shape))
        return color_clustered_image, color_index_map
   
    def __expand_image(self, image, size):
        """Expand image size

        Parameters
        ----------
        image : np.ndarray
            Input image
        size : int
            Size that want to expand image.
            If you want to guess the proper size, set size value under 1.

        Returns
        ----------
        output : np.ndarray
            Expanded image
        """
        STANDARD_SIZE_OF_IMAGE = 5000

        if size < 1:
            max_image_length = max(image.shape[1], image.shape[0])
            size = (STANDARD_SIZE_OF_IMAGE // max_image_length) + 1

        output = cv2.resize(image, None, fx = size, fy = size, interpolation = cv2.INTER_LINEAR)
        return output
    
    

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
    

if __name__ == "__main__":
    # How to Use?
    img = cv2.imread("./libs/lala.jpg")
    painting = Painting(img)
    painting_image, color_index_map = painting.run(
                                                k = 8,
                                                is_upscale = True,
                                                size = 2,
                                                blurring = True)
    
    drawing = LineDrawing(color_index_map)
    line_drawn_image = drawing.run(outline = True)
    cv2.imwrite("./libs/lala-after-line-drawn.jpg", line_drawn_image)
    