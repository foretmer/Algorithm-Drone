import richuru

richuru.install()

import matplotlib.pyplot as plt
import math

from loguru import logger
from roles import *


class Draw:
    def __init__(self, Drop_points: list[Dot], Depots: list[Dot], Drone_path: list[Drone], round):
        self.Drop_points = Drop_points
        self.Depots = Depots
        self.Drone_path = Drone_path
        self.x_depot = []
        self.y_depot = []
        self.x_drop_point = []
        self.y_drop_point = []
        self.round = round
        self.one_round_distance = 0

    def one_drone_distance(self, route: list[tuple]) -> float:
        total_distance = 0

        for i in range(len(route) - 1):
            x1, y1 = route[i]
            x2, y2 = route[i + 1]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_distance += distance

        return total_distance

    def draw_map(self):
        self.x_depot = [depot.x for depot in self.Depots]
        self.y_depot = [depot.y for depot in self.Depots]
        depot_index = [depot.id for depot in self.Depots]
        plt.scatter(self.x_depot, self.y_depot, marker="o", label="Depots")
        for i in range(len(depot_index)):
            plt.annotate(depot_index[i], (self.x_depot[i], self.y_depot[i]), textcoords="offset points",
                         xytext=(0, 10), ha='center')

        self.x_drop_point = [drop_point.x for drop_point in self.Drop_points]
        self.y_drop_point = [drop_point.y for drop_point in self.Drop_points]
        drop_point_index = [drop_point.id for drop_point in self.Drop_points]
        plt.scatter(self.x_drop_point, self.y_drop_point, marker="x", label="Drop_points")
        for i in range(len(drop_point_index)):
            plt.annotate(drop_point_index[i], (self.x_drop_point[i], self.y_drop_point[i]), textcoords="offset points",
                         xytext=(0, 5), ha='center')

    def draw_route(self) -> float:
        self.draw_map()
        for drone in self.Drone_path:
            route_x = [self.x_depot[drone.start_depot - 1]]
            route_y = [self.y_depot[drone.start_depot - 1]]

            for attr in drone.route:
                route_x.append(self.x_drop_point[attr.id - 1])
                route_y.append(self.y_drop_point[attr.id - 1])
            route_x.append(self.x_depot[drone.start_depot - 1])
            route_y.append(self.y_depot[drone.start_depot - 1])

            plt.plot(route_x, route_y)

            route = list(zip(route_x, route_y))
            round_distance = self.one_drone_distance(route)
            self.one_round_distance += round_distance
            
            logger.info(f"Drone {drone.drone_id} Round distance: {round_distance} km")
            if round_distance > 20:
                logger.warning(f"Drone {drone.drone_id} out of gas")

        plt.legend()

        plt.savefig(f"results/current_time_{self.round}_drone_{drone.drone_id}.png")
        plt.clf()

        self.round += 1
        return self.one_round_distance

