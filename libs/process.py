import cv2
import numpy as np

class Painting:
    """Change image to painting image.
    
    Parameters
    ----------
    img_path : str (default: None)
        File path of image

    Attributes
    ----------
    original_img : np.array
        Input original image
    painting : np.array
        Color clustered image (Applied K-Means Algorithm)
    colors : np.array
        Clustered color data list
    """

    def __init__(self, img_path):
        self.original_img = cv2.imread(img_path) 
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
            radius = 10, 
            sigma_color = 20,
            median_value = 5,
            step = 0):
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
        radius : int, optional (default: 10)
            bilateralFilter Parameter
        sigma_color : int, optional (default: 20)
            bilateralFilter Parameter
        median_value : int, optional (default: 5)
            medianBlur Parameter
        step : int, optional (default: 0)
            Blurring intensity by step size

        Returns
        ----------
        self.painting : np.ndarray
            Color clustered image
        """

        if blurring:
            target_image = self.blurring(div = div,
                                         radius = radius,
                                         sigma_color = sigma_color,
                                         median_value = median_value,
                                         step = step)
        else:
            target_image = self.original_img.copy()

        if is_upscale:
            target_image = self.__expand_image(target_image, size = size)

        self.painting, sse = self.__cluster_color_with_kmeans(target_image, 
                                                            number_of_color = k, 
                                                            attempts = attempts)
        return self.painting
    
    def blurring(self, 
                div,
                radius, 
                sigma_color,
                median_value,
                step):
        """Image blurring

        Parameters
        ----------
        div : int
            Reducing numbers of color on image
        radius : int
            bilateralFilter Parameter
        sigma_color : int
            bilateralFilter Parameter
        median_value : int
            medianBlur Parameter
        step : int
            Blurring intensity by step size

        Returns
        -------
        blurring : np.ndarray
            blurred Image
        """
        
        qimg = self.original_img.copy() # copy original image
        
        step = min(max(0, step), 5) # 1 <= step <= 5
        
        size_of_image = int( (qimg.shape[1] * qimg.shape[0]) ** 0.5 ) // 100
        sigma_color += min( int(size_of_image * 2.5), 90) + step * 4
        radius += min( int(size_of_image * 1.5), 40) + step * 2
        
        # blurring
        blurring = cv2.bilateralFilter(qimg, radius, sigma_color, 60)
        blurring = cv2.medianBlur(blurring, median_value)
        
        # reduce numbers of color
        blurring = blurring // div * div + div // 2
        return blurring
    
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
        sse : float
            Sum of squared error
        """
        
        height, width = image.shape[:2]
        training_data_samples = np.zeros([height * width, 3], dtype = np.float32)
        
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
        
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        self.colors = centers

        # for returns
        sse = round(sse ** 0.5 // 10, 2)
        color_clustered_image = res.reshape((image.shape))
        return color_clustered_image, sse
   
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
    
    
if __name__ == "__main__":
    # How to Use?
    painting = Painting( "./imagePath/image.jpg")
    result_image = painting.run(
                                k = 8,
                                is_upscale = True,
                                size = 2,
                                blurring = True)
    pass
    