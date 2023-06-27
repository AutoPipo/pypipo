# -*- coding: utf-8 -*-

def convert_image(filepath, **kwargs):
    print('=== convert image ===')
    print(filepath)
    print(kwargs)
    #######
    '''
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
    '''
    return 

