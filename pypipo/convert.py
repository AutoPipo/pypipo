# -*- coding: utf-8 -*-

from libs.process import Painting, LineDrawing

def convert_image(filepath, outputpath, **kwargs):
    print('=== convert image ===')
    
    painting = Painting(filepath)
    p, color_index_map = painting.run(**kwargs)
    
    drawing = LineDrawing(color_index_map)
    line_drawn_image = drawing.run(outline = True)
    drawing.save(outputpath, line_drawn_image)
    
    return 

