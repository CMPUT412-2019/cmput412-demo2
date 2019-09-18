import networkx as nx
import numpy as np
from PIL import Image


class Map:
    def __init__(self, origin, scale=0.1, width=100, height=100):  # type: (np.ndarray, float, int, int)
        self.world_origin = origin
        self.grid_origin = np.array([width, height]) / 2
        self.scale = scale
        self.width = width
        self.height = height

        self.grid = np.zeros((self.width, self.height), dtype=np.bool)

    def _world_to_grid(self, position, round=True):  # type: (np.ndarray, bool) -> np.ndarray
        position = (position - self.world_origin) / self.scale + self.grid_origin
        return np.round(position) if round else position

    def _grid_to_world(self, position, center=True):  # type: (np.ndarray, bool) -> np.ndarray
        offset = self.scale/2 if center else 0
        return (position - self.grid_origin + offset) * self.scale + self.world_origin

    def _world_distance_from(self, position):  # type: (np.ndarray) -> np.ndarray
        xx, yy = np.mgrid[:self.width, :self.height]
        xx, yy = self._grid_to_world(np.stack((xx, yy), axis=-1)).transpose((2, 0, 1))
        return np.sqrt((xx - position[0]) ** 2 + (yy - position[1]) ** 2)

    def _grid_distance_from(self, position):  # type: (np.ndarray) -> np.ndarray
        xx, yy = np.mgrid[:self.width, :self.height]
        return (xx - position[0]) ** 2 + (yy - position[1]) ** 2

    def _grid_closest_free(self, position):  # type: (np.ndarray) -> np.ndarray
        d = self._grid_distance_from(position)
        return np.unravel_index(np.argmin(d + self.grid * 10000000), d.shape)

    def fill_rect(self, mins, maxs, fillval=True):  # type: (np.ndarray, np.ndarray, bool) -> None
        grid_mins, grid_maxs = self._world_to_grid(mins), self._world_to_grid(maxs)
        self.grid[grid_mins[0]:grid_maxs[0], grid_mins[1]:grid_maxs[1]] = fillval

    def fill_circle(self, center, radius, fillval=1):  # type: (np.ndarray, float, bool) -> None
        self.grid[self._world_distance_from(center) < radius] = fillval

    def image(self):  # type: () -> Image
        return Image.fromarray(self.grid.astype(np.uint8) * 255, 'L')

    def pathfind(self, start, end):  # type: (np.ndarray, np.ndarray) -> List[np.ndarray]

        # see https://stackoverflow.com/questions/55772715/how-to-create-8-cell-adjacency-map-for-a-diagonal-enabled-a-algorithm-with-the
        G = nx.generators.grid_2d_graph(self.width, self.height)  # type: nx.Graph
        G.add_edges_from([
             ((x, y), (x + 1, y + 1))
             for x in range(self.width - 1)
             for y in range(self.height - 1)
         ] + [
             ((x + 1, y), (x, y + 1))
             for x in range(self.width - 1)
             for y in range(self.height - 1)
         ], weight=1.4)

        obstacles = list((x, y) for x in range(self.width) for y in range(self.height) if self.grid[x, y])
        G.remove_nodes_from(obstacles)
        start_node = tuple(self._grid_closest_free(self._world_to_grid(start).astype(int)))
        end_node = tuple(self._grid_closest_free(self._world_to_grid(end).astype(int)))
        path = nx.algorithms.shortest_paths.astar_path(G, start_node, end_node, lambda a, b: np.linalg.norm(np.array(a) - np.array(b)))
        return [start] + list(self._grid_to_world(np.asarray(node)) for node in path) + [end]

    def path_image(self, path, include_grid=True):  # type: (List[np.ndarray], bool) -> Image
        img = np.copy(self.grid) if include_grid else np.zeros_like(self.grid)
        for (x, y) in path:
            x, y = self._world_to_grid(x), self._world_to_grid(y)
            img[x, y] = 1
        return Image.fromarray(img.astype('uint8') * 255, 'L')


if __name__ == '__main__':
    m = Map(np.array([0, 0]), scale=1)
    m.fill_circle(np.array([0, 0]), 20)
    path = m.pathfind(np.array([-10, -10]), np.array([30, 30]))
    m.path_image(path, include_grid=False).show()
