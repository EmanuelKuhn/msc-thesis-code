from typing import Dict, List, Tuple

import numpy as np
import rasterio.features
from shapely import geometry, wkt

from rasterio.features import shapes, rasterize

from collections import defaultdict
from itertools import pairwise


# RPLAN helpers
def get_room_shapes(instance_mask) -> Dict[int, geometry.Polygon]:
    room_polygons = {}

    for shape, value in rasterio.features.shapes(instance_mask.astype(np.int16), mask=(instance_mask > 0)):
        poly = geometry.shape(shape)

        room_polygons[int(value)] = poly

    return room_polygons


def dilate_rooms(room_polygons: Dict[int, geometry.Polygon]):
    room_polygons = room_polygons.copy()

    # First compute the minimum distance between rooms
    mp_all_rooms = geometry.MultiPolygon(room_polygons.values())

    clearance = mp_all_rooms.minimum_clearance

    # Dilate each room to occupy half of the wall
    dilation_amount = clearance / 2

    room_polygons_dilated = {}

    for i, room in room_polygons.items():
        room_dilated = room.buffer(dilation_amount, join_style=geometry.JOIN_STYLE.mitre)

        room_polygons_dilated[i] = room_dilated

    return room_polygons_dilated


def get_room_categories(room_polygons, category_map):

    room_categories = {}

    for room_id, room_polygon in room_polygons.items():
        room_categories[room_id] = get_room_category(room_polygon, category_map)
    
    return room_categories

def get_room_category(room_polygon, category_map):
    """Method to get the category of a room polygon."""
    
    # Room mask for the target room
    room_mask = rasterize([room_polygon], out_shape=(category_map.shape))

    # Loop over the category shapes that are inside the room mask. Select the largest one as the room category.
    max_area = 0
    category = None
    
    for proposal_shape, proposal_category in shapes(category_map, mask=room_mask):
        if geometry.shape(proposal_shape).area > max_area:
            max_area = geometry.shape(proposal_shape).area
            category = proposal_category

    return category



def compute_wall_thickness(room_polygons):
    # Compute the minimum distance between rooms
    mp_all_rooms = geometry.MultiPolygon(room_polygons.values())

    return mp_all_rooms.minimum_clearance


### Door parsing

def linestring_to_sorted_wkt(linestring: geometry.LineString):
    """Utility method for representing a linestring in a canonical way, for hashing purposes."""
    return geometry.LineString(sorted(linestring.coords)).wkt

def parse_doors(room_polygons, category_map, door_length_threshold=3):
    """Method that parses doors.
        
    Args:
        room_polygons (Dict[int, geometry.Polygon]): Dictionary mapping room ids to room polygons.
        category_map (np.ndarray): rplan_data.category.
    """

    doors = []

    for shape, val in shapes(category_map, mask=(category_map==17) | (category_map==15)):
        doors.append(geometry.shape(shape))
    
    intersections = set()

    door_to_wall = defaultdict(list)

    for room_id, room in room_polygons.items():
        for wall_index, wall in enumerate(pairwise(room.exterior.coords)):

            wall = geometry.LineString(wall)

            for door in doors:
                intersection = door.intersection(wall)

                if not intersection.is_empty:

                    if isinstance(intersection, geometry.LineString):
                        door_linestring_wkt = linestring_to_sorted_wkt(intersection)

                        intersections.add(door_linestring_wkt)

                        door_to_wall[door_linestring_wkt].append((room_id, wall_index))
    
    doors_dict = {index: door for index, door in enumerate(intersections)}

    doors_dict = {index: wkt.loads(door) for index, door in doors_dict.items()}

    doors_dict = {id: line for id, line in doors_dict.items() if line.length > door_length_threshold}

    return doors_dict



# This method should be called again after changing room geometry.
def compute_door_to_rooms(room_polygons, doors_dict, wall_buffer=0.5) -> Dict[int, List[int]]:
    """Find the rooms that a door intersects.
    
    Args:
        room_polygons (Dict[int, geometry.Polygon]): Dictionary mapping room ids to room polygons.
        category_map (np.ndarray): rplan_data.category.
        wall_buffer (float, optional): Amount to buffer walls by before checking if a door intersects them.
        door_length_threshold (float, optional): Minimum length of a door to be considered a door.
    """

    door_to_rooms = defaultdict(list)

    for room_id, room in room_polygons.items():
        walls_buffered = room.buffer(wall_buffer, join_style=2, cap_style=2)

        for door_id, door in doors_dict.items():
            if door.intersects(walls_buffered):
                door_to_rooms[door_id].append((room_id, door.intersection(walls_buffered).length))
    
    for door_id, intersections in door_to_rooms.items():

        intersections = sorted(intersections, key=lambda x: x[1], reverse=True)

        door_to_rooms[door_id] = [x[0] for x in intersections]

        if len(door_to_rooms[door_id]) > 2:
            door_to_rooms[door_id] = door_to_rooms[door_id][:2]
        
        
    return door_to_rooms
