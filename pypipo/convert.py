# -*- coding: utf-8 -*-

from libs.process import Painting, LineDrawing

def convert_image(filepath, outputpath, **kwargs):
    print('=== convert image ===')
    
    painting = Painting(filepath)
    painting_img, color_index_map = painting.run(**kwargs)
    colorNames_, colors_ = painting.getColorFromImage(painting_img)

    
    drawing = LineDrawing(color_index_map)
    line_drawn_image = drawing.run(outline = True)
    img_lab, lab = drawing.getImgLabelFromImage(colors_, painting_img)
    drawing.save(outputpath, line_drawn_image)
    
    return 

