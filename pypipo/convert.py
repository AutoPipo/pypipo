# -*- coding: utf-8 -*-

from libs.process import Painting, LineDrawing, ColorspaceIndexing
from libs.utils import *

def convert_image(filepath, outputpath, **kwargs):
    print('=== convert image ===')
    print(outputpath)
    
    painting = Painting(filepath)
    painting_img, color_index_map = painting.run(**kwargs)
    # TODO: change values name
    color_indexs, color_rbg_values = painting.get_clustered_color_info(painting_img)

    
    drawing = LineDrawing(color_index_map)
    line_drawn_image = drawing.run(outline = True)
    # TODO: change values name
    img_lab, lab = drawing.get_image_lab(color_rbg_values, painting_img)

    print('=== Numbering ===')
    numbering = ColorspaceIndexing(painting_img, line_drawn_image, color_indexs, color_rbg_values)
    # TODO: add cli, color_label parameter
    output = numbering.run(img_lab, lab, color_label = True)
    img_save(outputpath, output)
    
    return 

