from itertools import pairwise
import warnings
from matplotlib import pyplot as plt
from shapely.geometry import CAP_STYLE
from PIL import Image, ImageDraw

import rplanpy

from .vectorize import get_room_shapes, dilate_rooms, compute_wall_thickness

from . import vectorize

import networkx as nx

from shapely import geometry, ops

from .rendering import DrawingContext, random_color

from typing import List, Dict

import pickle as pkl

import numpy as np


# Class to encapsulate a floorplan (vectorize, render it, compute the graph)
class FloorPlan:

    # A version number that indicates which major revision of the FloorPlan class a pickled floorplan is compatible with.
    CURRENT_PICKLE_VERSION = 2

    """An Rplan floorplan
    
        Attributes:
            room_polygons: A dictionary mapping room ids to polygons
            room_categories: A dictionary mapping room ids to categories (ints as defined in the Rplan dataset)
            doors_dict: A dictionary mapping door ids to lines representing the door
            door_to_rooms: A dictionary mapping door ids to a list of room ids that share the door
            wall_thickness: The thickness of the walls in the underlying floorplan
            graph: A graph of the rooms, where each node is a room id and each nodes are connected if they share a door
    """
    def __init__(self, 
                 room_polygons: Dict[int, geometry.Polygon], 
                 room_categories: Dict[int, int],
                 doors_dict: Dict[int, geometry.LineString], 
                 door_to_rooms: Dict[int, List[int]], 
                 wall_thickness, 
                 validate_graph=False
                ) -> None:


        self.room_polygons = room_polygons

        self.room_categories = room_categories

        self.doors_dict = doors_dict
        self.door_to_rooms = door_to_rooms

        self.wall_thickness = wall_thickness

        self.doors_graph = self.compute_doors_graph()

        if validate_graph:
            # Make sure the room graph is connected
            self.validate_doors_graph()

    def validate_doors_graph(self, throw_invalid=True) -> bool:
        is_connected = nx.is_connected(self.doors_graph)

        if throw_invalid:
            assert is_connected, f"The room graph was not connected: {self.doors_graph.edges} ({self.doors_graph.nodes})"

        return is_connected

    def save_pickle(self, path):

        with open(path, "wb") as f:
            f.write(self.to_pickle_str())
        
        return path
    
    def to_pickle_str(self):
        data = {
            "pickle_version": FloorPlan.CURRENT_PICKLE_VERSION,
            "room_polygons": self.room_polygons,
            "room_categories": self.room_categories,
            "doors_dict": self.doors_dict,
            "door_to_rooms": self.door_to_rooms,
            "wall_thickness": self.wall_thickness
        }

        return pkl.dumps(data)
    
    @staticmethod
    def load_pickle(path, validate_graph=True):
        with open(path, "rb") as f:
            return FloorPlan.from_pickle(f.read(), validate_graph=validate_graph)

    @staticmethod
    def from_pickle(pickle_str, validate_graph=False):
        data = pkl.loads(pickle_str)

        assert data["pickle_version"] == FloorPlan.CURRENT_PICKLE_VERSION, f"Unexpected pickle version: {data['pickle_version']}"

        return FloorPlan(
            data["room_polygons"],
            data["room_categories"],
            data["doors_dict"],
            data["door_to_rooms"],
            data["wall_thickness"],
            validate_graph=validate_graph
        )

    @staticmethod
    def from_rplan_data(rplan_data: rplanpy.data.RplanData, validate_graph=True, parse_doors=True):
        room_polygons = get_room_shapes(rplan_data.instance)

        room_categories = vectorize.get_room_categories(room_polygons, rplan_data.category)

        original_wall_thickness = compute_wall_thickness(room_polygons)

        # Some rplan samples behave weirdly, in that minimum_clearance > 0 after dilating rooms
        while compute_wall_thickness(room_polygons) > 0:
            # print(f"clearance: {self.debug_shapes.minimum_clearance}")
            room_polygons = dilate_rooms(room_polygons)
            original_wall_thickness += compute_wall_thickness(room_polygons)

        if parse_doors:
            doors_dict = vectorize.parse_doors(room_polygons, rplan_data.category)

            door_to_rooms = vectorize.compute_door_to_rooms(room_polygons, doors_dict)
        else:
            doors_dict, door_to_rooms = None, None

        return FloorPlan(room_polygons, room_categories, doors_dict, door_to_rooms, original_wall_thickness, validate_graph=validate_graph)


    @property
    def doors(self):
        return geometry.MultiLineString(list(self.doors_dict.values()))

    @property
    def front_door(self):
        return self.doors.intersection(self.exterior)

    def compute_doors_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.room_polygons.keys())

        # Edges are rooms that share a door
        edges = [rooms for rooms in self.door_to_rooms.values() if len(rooms) == 2]

        G.add_edges_from(edges)

        return G
    
    @property
    def room_topology_graph(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return self.compute_room_topology_graph(minimum_shared_edge_length=5)

    def compute_room_topology_graph(self, minimum_shared_edge_length):
        
        G = nx.Graph()
        G.add_nodes_from(self.room_polygons.keys())

        for room1_id, room1 in self.room_polygons.items():
            for room2_id, room2 in self.room_polygons.items():
                if room1_id == room2_id:
                    continue

                intersection = room1.intersection(room2)

                if intersection.length > minimum_shared_edge_length:
                    G.add_edge(room1_id, room2_id)

        return G


    @property
    def all_rooms(self):
        """Returns a MultiPolygon of all rooms"""
        return geometry.MultiPolygon(self.room_polygons.values())

    @property
    def exterior(self):
        # """Returns the exterior of the floor plan as a Polygon"""
        try:
            return ops.unary_union(self.all_rooms).exterior
        except:
            raise ValueError("exterior: Couldn't merge rooms into single polygon")

    def get_wall(self, room, wall_index):
        wall = list(pairwise(self.room_polygons[room].exterior.coords))[wall_index]

        return geometry.LineString(wall)


    def get_passages(self, subtraction_distance=5):
        """Method that returns a list of wall elements that are passages.
        
        Passages are walls between two rooms that are share a door in the rplan dataset. If two rooms share multiple wall segments,
        they are joined into a single passage. The passages are not necessarily MultiLineStrings, because they are computed as the intersection
        of two rooms.

        In order to leave a gap between multiple passages, other rooms buffered by `subtraction_distance` are subtracted from the passage.

        Args:
            subtraction_distance (int, optional): Distance buffer other walls by before subtracting them from a passage.
        """

        passages = []

        for rooms in self.door_to_rooms.values():
            if len(rooms) == 2:
                rooms = [self.room_polygons[room] for room in rooms]

                other_rooms = self.all_rooms - geometry.MultiPolygon(rooms)

                passage = rooms[0].intersection(rooms[1])

                passage = passage - other_rooms.buffer(subtraction_distance, cap_style=2, join_style=2)

                if not passage.is_empty and (not isinstance(passage, geometry.Point)):
                    passages.append(passage)
        
        return passages


    def render(self, 
               plot_doors=False, 
               plot_passages=False,
               plot_categories=False,
               wall_thickness=None,

               plot_walls=True,

               equal_wall_thickness=False,

               # Debug options
               debug_output=False, 
               highlight_room=None, 
               background_color=(255, 255, 255, 255),
               highlight_wall=0) -> Image.Image:
        
        d_ctx = DrawingContext(scale_factor=1)
        
        # print(f"{plot_categories=}")
        # print(f"{plot_passages=}")

        if wall_thickness is None:
            wall_thickness = self.wall_thickness

        inner_wall_thickness = wall_thickness if equal_wall_thickness else wall_thickness / 3

        img = Image.new("RGBA", (256, 256), background_color)

        inner_walls = self.all_rooms.boundary.buffer(inner_wall_thickness / 2, cap_style=2, join_style=2)
        outer_wall = (self.exterior).buffer(wall_thickness / 2, cap_style=2, join_style=2)


        if plot_categories:
            for room_id, category in self.room_categories.items():

                room = self.room_polygons[room_id]

                img =d_ctx.draw_polygon(img, room, fill=tuple(rplanpy.utils.ROOM_COLOR[category]))

        if plot_walls:
            img =d_ctx.draw_polygon(img, inner_walls, fill=tuple(rplanpy.utils.ROOM_COLOR[16]))

        draw = ImageDraw.Draw(img)

        if plot_passages:

            passage_color = (240, 160, 100, 255)

            all_passages = ops.unary_union(self.get_passages()).buffer(inner_wall_thickness / 2, cap_style=2, join_style=2)

            img =d_ctx.draw_polygon(img, all_passages, fill=passage_color)

        if plot_walls:
            img =d_ctx.draw_polygon(img, outer_wall, fill=tuple(rplanpy.utils.ROOM_COLOR[14]))


        if plot_doors:

            door_color = tuple(rplanpy.utils.ROOM_COLOR[17])

            # door_color = (240, 160, 100, 255)


            all_doors = ops.unary_union(self.doors).buffer(inner_wall_thickness / 2, cap_style=2, join_style=2)

            img =d_ctx.draw_polygon(img, all_doors, fill=door_color)

            front_door = self.front_door.buffer(wall_thickness / 2, cap_style=2, join_style=2)

            img = d_ctx.draw_polygon(img, front_door, fill=door_color)


        draw = ImageDraw.Draw(img)


        if debug_output:

            for room_id, room in self.room_polygons.items():
                point = room.representative_point()

                draw.text(point.coords, f"{room_id}", fill=(10, 10, 0))
            
        if highlight_room:
            room = self.room_polygons[highlight_room]
            
            wall = self.get_wall(highlight_room, highlight_wall)            
            next_wall = self.get_wall(highlight_room, (highlight_wall + 1) % (len(room.exterior.coords) - 1))

            draw.line(list(wall.coords), fill=(255, 0, 100), width=2)
            # draw.line(list(next_wall.coords), fill=(0, 0, 255), width=2)




        img = Image.alpha_composite(Image.new("RGBA", (256, 256), (255, 255, 255, 255)), img)
        img = img.convert("RGB")


        return img

    
    def render_overlay_graph(self, plot_topology=True, plot_doors_graph=True, plot_img=True, **render_kwargs):
        """Renders the floor plan and overlays the computed graph of rooms and room connections."""

        # By default, render which wall segments are passages
        if "plot_passages" not in render_kwargs:
            render_kwargs["plot_passages"] = True

        img = self.render(**render_kwargs)

        positions = {id: np.array(room.representative_point().coords)[0] for id, room in self.room_polygons.items()}
        
        plt.imshow(img, alpha=1 if plot_img else 0)

        if plot_topology:
            nx.draw(self.room_topology_graph, with_labels=False, pos=positions, edge_color="grey")
        
        if plot_doors_graph:
            nx.draw(self.doors_graph, with_labels=True, pos=positions, edge_color="black", width=2)
        
    
    def render_random_rects(self):
        """Method that should have been called: `render_random_colors`.
        
        Renders a floorplan with random colors for each room. The colors really are random, 
        and drawn from a uniform distribution of 256^3 colors."""

        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))

        d_ctx = DrawingContext()

        for room in self.room_polygons.values():
            img =d_ctx.draw_polygon(img, room, fill=random_color(4))

        return img.convert("RGB")
    
    def compute_bounding_box(self):
        return self.exterior.bounds

    def _repr_svg_(self):
        return self.all_rooms._repr_svg_()
