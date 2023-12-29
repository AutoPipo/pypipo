import cv2
from pypipo.convert import pipo_convert, pipo_convert_dev

def sample_of_pypipo():
    number_of_repeat = 3
    for i in range(number_of_repeat):
        # sample_image_binary = cv2.imread(f'./pypipo/sample/sample-image.png')
        
        if i == 0:
            sample_image_binary = cv2.imread(f'./pypipo/sample/image-0.png')
            pipo_output_binary = pipo_convert_dev("./pypipo/sample/", 
                                          sample_image_binary, 
                                          number = 20,
                                          is_upscale = True, 
                                          target_size = 2)
        else:
            sample_image_binary = cv2.imread(f'./pypipo/sample/image-{i}.png')
            pipo_output_binary = pipo_convert_dev("./pypipo/sample/", 
                                          sample_image_binary, 
                                          number = 16, cnt = i)
    return 

if __name__ == "__main__":
    sample_of_pypipo()
