import cv2
from pypipo.convert import pipo_convert, pipo_convert_dev

def sample_of_pypipo():
    sample_image_binary = cv2.imread(f'./pypipo/sample/sample-image.png')
    pipo_output_binary = pipo_convert_dev("./pypipo/sample/", 
                                          sample_image_binary, 
                                          is_upscale = True, 
                                          target_size = 2)

    return 

if __name__ == "__main__":
    sample_of_pypipo()
