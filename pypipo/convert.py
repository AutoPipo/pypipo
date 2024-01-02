# -*- coding: utf-8 -*-

from pypipo.libs.process import *
from pypipo.libs.utils import *

# for release
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

# for dev
def pipo_convert_dev(output_path, input_color_image, color_label = True, is_dev = True, cnt = 0, **kwargs):
    painting = Painting(input_color_image)
    painting_img, color_index_map = painting.run(**kwargs)
    color_indexs, color_rbg_values = painting.get_clustered_color_info(painting_img)
    img_save(f"{output_path}/sample-painting.png", painting_img)
    
    drawing = LineDrawing(color_index_map)
    line_drawn_image = drawing.run(outline = True)
    img_save(f"{output_path}/sample-lined.png", line_drawn_image)
    img_lab, lab = drawing.get_image_lab(color_rbg_values, painting_img)

    numbering = ColorspaceIndexing(painting_img, line_drawn_image, color_indexs, color_rbg_values, is_dev)
    output = numbering.run(img_lab, lab, color_label = color_label)
    img_save(f"{output_path}/sample-output.png", output)

    img_save(f"{output_path}/sample-painted-output.png", numbering.dev_get_result_image())
    img_save(f"{output_path}/image-{cnt+1}.png", numbering.dev_get_result_image())

    return output
