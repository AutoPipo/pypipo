# -*- coding: utf-8 -*-

from pypipo.libs.process import Painting, LineDrawing, ColorspaceIndexing
from pypipo.libs.utils import *

def pipo_convert(input_color_image, color_label = True, **kwargs):
    painting = Painting(input_color_image)
    painting_img, color_index_map = painting.run(**kwargs)
    color_indexs, color_rbg_values = painting.get_clustered_color_info(painting_img)
    
    drawing = LineDrawing(color_index_map)
    line_drawn_image = drawing.run(outline = True)
    # TODO: change values name : img_lab, lab
    img_lab, lab = drawing.get_image_lab(color_rbg_values, painting_img)

    numbering = ColorspaceIndexing(painting_img, line_drawn_image, color_indexs, color_rbg_values)
    output = numbering.run(img_lab, lab, color_label = color_label)
    return output

