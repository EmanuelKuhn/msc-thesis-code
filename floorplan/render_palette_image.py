
import sys
sys.path.append("..")


import typing
from shapely import affinity, geometry, ops

from PIL import Image, ImageDraw

import numpy as np

import rplanpy


MODE_GRAYSCALE_ALPHA = "LA"

class DrawingContextGrayScale:
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor
    
    def draw_polygon(self, img: Image.Image, polygon, fill: typing.Tuple[int, int]) -> Image.Image:

        if self.scale_factor != 1.0:
            polygon = affinity.scale(polygon, xfact=self.scale_factor, yfact=self.scale_factor, origin=(0, 0))

        if isinstance(polygon, geometry.MultiPolygon):
            for p in polygon.geoms:
                img = self.draw_polygon(img, p, fill)
            
            return img

        new_layer = Image.new(MODE_GRAYSCALE_ALPHA, img.size)

        draw = ImageDraw.Draw(new_layer)

        draw.polygon(polygon.exterior.coords, fill=fill)

        for interior in polygon.interiors:
            draw.polygon(interior.coords, fill=(0, 0), outline=fill, width=1)
        
        return Image.alpha_composite(img.convert("RGBA"), new_layer.convert("RGBA")).convert("LA")


ROOM_CLASS = {
    0: "Living room",
    1: "Master room",
    2: "Kitchen",
    3: "Bathroom",
    4: "Dining room",
    5: "Child room",
    6: "Study room",
    7: "Second room",
    8: "Guest room",
    9: "Balcony",
    10: "Entrance",
    11: "Storage",
    12: "Walk-in",

# Not room classes:
    # 13: "External area",
    # 14: "Exterior wall",
    # 15: "Front door",
    # 16: "Interior wall",
    # 17: "Interior door",
}

BACKGROUND=13
WALL=14
PASSAGE=15

ROOM_CATEGORIES = {key: f"room category: {key} ({ROOM_CLASS[key]})" for key in range(0, 13)}

PALETTE_CATEGORIES = {
    **ROOM_CATEGORIES,
    WALL: "WALL",
    PASSAGE: "PASSAGE",
    BACKGROUND: "BACKGROUND",
}

assert len(PALETTE_CATEGORIES) == 16, f"There should be 16 categories, but there are {len(PALETTE_CATEGORIES)}"

def rplan_palette_tuples():
    colors = rplanpy.utils.ROOM_COLOR

    # colors[12] walkin was the color of walls instead of storage
    colors[12] = colors[11]

    palette = [rplanpy.utils.ROOM_COLOR[i] for i in range(13)]

    palette += [rplanpy.utils.ROOM_COLOR[13]]
    palette += [(0, 0, 0)] # WALL COLOR
    palette += [(200, 100, 10)] # PASSAGE COLOR

    palette += [rplanpy.utils.ROOM_COLOR[13]] # MASKED

    return palette

def make_pil_palette(palette_tuples):
    palette = np.array(palette_tuples).reshape(-1).tolist()

    return palette

PALETTE_ID_OFFSET = 100

def rplan_palette_for_room_ids(room_categories_dict, id_offset=PALETTE_ID_OFFSET, base_pallette=rplan_palette_tuples()):
    maximum_room_id = max(room_categories_dict.keys(), key=lambda x: int(x))

    palette = [(0, 0, 0)] * (id_offset + maximum_room_id + 2)

    for i in [13, 14, 15, 15]:
        palette[i] = base_pallette[i]
    
    for room_id, room_category in room_categories_dict.items():
        room_id = int(room_id)

        room_category = int(room_category)

        assert room_id + id_offset < 200, f"Exptect room_id {room_id} + id_offset {id_offset} to be less than 200"
        palette[room_id + id_offset] = base_pallette[room_category]
    

    return palette




# Renders walls and background with mapping definded above, and roomids as room_id + PALETTE_ID_OFFSET.
# Then set a palette that shows correct colors for walls and maps room_ids + PALETTE_ID_OFFSET to color of it's category.
# This is usefull for later being able to mask individual rooms.
def render_room_number_palette_image(fp, wall_thickness, plot_passages, plot_room_ids=True, put_palette=True) -> Image.Image:
    BACKGROUND_COLOR = (BACKGROUND, 255)
    WALL_COLOR = (WALL, 255)
    PASSAGES_COLOR = (PASSAGE, 255)
    
    d_ctx = DrawingContextGrayScale(scale_factor=1)
    
    img = Image.new(MODE_GRAYSCALE_ALPHA, (256, 256), BACKGROUND_COLOR)

    inner_wall_thickness = wall_thickness / 2

    inner_walls = fp.all_rooms.boundary.buffer(inner_wall_thickness / 2, cap_style=2, join_style=2)
    outer_wall = (fp.exterior).buffer(wall_thickness / 2, cap_style=2, join_style=2)


    if plot_room_ids:
        for room_id, category in fp.room_categories.items():

            category = int(category)

            assert category in ROOM_CATEGORIES, f"Category {category} is not a valid room category"

            room = fp.room_polygons[room_id]

            img = d_ctx.draw_polygon(img, room, fill=(PALETTE_ID_OFFSET + int(room_id), 255))

    
    img = d_ctx.draw_polygon(img, inner_walls, fill=WALL_COLOR)

    if plot_passages:

        passage_color = PASSAGES_COLOR
        all_passages = ops.unary_union(fp.get_passages()).buffer(inner_wall_thickness / 2, cap_style=2, join_style=2)

        img = d_ctx.draw_polygon(img, all_passages, fill=passage_color)

    img = d_ctx.draw_polygon(img, outer_wall, fill=WALL_COLOR)

    # img = Image.alpha_composite(Image.new("LA", (256, 256), BACKGROUND_COLOR), img)
    img = img.convert("P")

    if put_palette:
        palette = make_pil_palette(rplan_palette_for_room_ids(fp.room_categories, id_offset=PALETTE_ID_OFFSET))

        img.putpalette(palette)

    img.format = "PNG"

    return img



def mask_ids(palette_img: np.array, ids: typing.List[int], masked_value) -> np.array:
    for id in ids:
        palette_img = np.where(palette_img == id, masked_value, palette_img)

    return palette_img

def mask_passages(palette_img: np.array, masked_value=WALL) -> np.array:
    return mask_ids(palette_img, [PASSAGE], masked_value)


# Fast way to get the indices of the img palette that have a non-zero color. Only returns indices greater than start.
def get_non_zero_colors(img, start):
    valid_room_ids = []

    palette = img.getpalette()

    for index in range(start, int(len(palette)/3)):
        triplet = palette[index * 3: index * 3 + 3]

        if triplet != [0, 0, 0]:
            valid_room_ids.append(index)

    return valid_room_ids


def get_all_ids_to_mask(img: Image):
    categories_to_mask = get_non_zero_colors(img, start=PALETTE_ID_OFFSET)

    return categories_to_mask

def get_random_ids_to_mask_uniform(img: Image):
    categories_to_mask = get_non_zero_colors(img, start=PALETTE_ID_OFFSET)
    
    categories_to_mask = np.random.choice(categories_to_mask, np.random.randint(0, len(categories_to_mask) + 1), replace=False)

    return categories_to_mask

def get_n_random_ids_to_mask(img: Image.Image, number_of_ids_to_mask):
    categories_to_mask = get_non_zero_colors(img, start=PALETTE_ID_OFFSET)

    number_of_ids_to_mask = min(number_of_ids_to_mask, len(categories_to_mask))
    
    categories_to_mask = np.random.choice(categories_to_mask, number_of_ids_to_mask, replace=False)

    return categories_to_mask


# room_masked_value is 13 by default, because that corresponds to the external area color.
def mask_room_id_img(img: Image.Image, ids_to_mask: list, should_mask_passages=False, room_masked_value=13) -> np.array:
    
    img_masked_array = np.array(img)
    img_masked_array = mask_ids(img_masked_array, ids_to_mask, room_masked_value)
    
    if should_mask_passages:
        img_masked_array = mask_passages(img_masked_array)

    img_masked = Image.fromarray(img_masked_array)

    img_masked.putpalette(img.palette)

    return img_masked

def set_palette_entry(img: Image.Image, index, color):

    palette = img.getpalette()

    palette[index * 3: index * 3 + 3] = color

    img.putpalette(palette)

class MaskRandomCategoriesAugmentation:
    def __init__(self, mask_none_pct: float=0.5, mask_all_pct: float=0.1, mask_uniform_pct: float=0.2, mask_one_pct: float=0.2, mask_passages_pct: float=0.25, masked_color: list|None=None) -> None:

        mask_pcts = {
            "none": mask_none_pct,
            "all": mask_all_pct,
            "uniform": mask_uniform_pct,
            "one": mask_one_pct
        }

        self.mask_types = list(mask_pcts.keys())
        self.mask_pcts = list(mask_pcts.values())

        self.mask_passages_pct = mask_passages_pct

        self.masked_color = masked_color


        # 99 should not be used yet in the palette, because the room colors start after 100. 13 if the color of the external area.
        self.masked_value = 13 if masked_color is None else 99
    
    def __call__(self, img: Image.Image) -> Image.Image:

        should_mask_passages = np.random.random() < self.mask_passages_pct

        mask_type = np.random.choice(self.mask_types, p=self.mask_pcts)

        if self.masked_color is not None:
            assert self.masked_value == 99, "Expected masked_value to be 99 when masked_color is not None."
            set_palette_entry(img, 99, self.masked_color)

        if mask_type == "none":
            return mask_room_id_img(img, [], should_mask_passages=should_mask_passages, room_masked_value=self.masked_value)
        elif mask_type == "all":
            return mask_room_id_img(img, get_all_ids_to_mask(img), should_mask_passages=should_mask_passages, room_masked_value=self.masked_value)
        elif mask_type == "uniform":
            return mask_room_id_img(img, get_random_ids_to_mask_uniform(img), should_mask_passages=should_mask_passages, room_masked_value=self.masked_value)
        elif mask_type == "one":
            return mask_room_id_img(img, get_n_random_ids_to_mask(img, 1), should_mask_passages=should_mask_passages, room_masked_value=self.masked_value)
        else:
            raise ValueError(f"Unknown mask type {mask_type}")
    
    @staticmethod
    def init_without_masking():
        return MaskRandomCategoriesAugmentation(mask_none_pct=1.0, mask_all_pct=0.0, mask_uniform_pct=0.0, mask_one_pct=0.0, mask_passages_pct=0.0)
    