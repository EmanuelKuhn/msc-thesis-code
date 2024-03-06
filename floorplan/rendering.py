import shapely.geometry as geometry

from PIL import Image, ImageDraw

import random

from shapely import affinity

class DrawingContext:
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor
    
    def draw_polygon(self, img: Image.Image, polygon, fill=(0, 0, 0, 255)) -> Image.Image:

        if self.scale_factor != 1.0:
            polygon = affinity.scale(polygon, xfact=self.scale_factor, yfact=self.scale_factor, origin=(0, 0))

        if isinstance(polygon, geometry.MultiPolygon):
            for p in polygon.geoms:
                img = self.draw_polygon(img, p, fill)
            
            return img

        new_layer = Image.new("RGBA", img.size)

        draw = ImageDraw.Draw(new_layer)

        draw.polygon(polygon.exterior.coords, fill=fill)

        for interior in polygon.interiors:
            draw.polygon(interior.coords, fill=(0, 0, 0, 0), outline=fill, width=1)
        
        return Image.alpha_composite(img, new_layer)




def random_color(n_channels=3):
    return tuple([random.randint(0, 255) for _ in range(n_channels)])