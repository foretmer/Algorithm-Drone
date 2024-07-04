from typing import List, Tuple
import random
import math

class Vehicle:
    def __init__(self, vehicle_id: int, capacity: int, speed: int ):
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.route = []
        self.load = 0
        self.speed = speed  # 1 unit distance per minute
        self.orders = []
        self.real_mileage = 0
        self.max_mileage = 20
        
class DropPoint:
    def __init__(self, x: float, y: float,id:int):
        self.x = x
        self.y = y
        self.id = id
        self.depot = None

class Order:
    def __init__(self, order_id: int,  destination: DropPoint, demand: int, time_window: Tuple[int, int], priority: int):
        self.order_id = order_id
        self.destination = destination
        self.demand = demand
        self.time_window = time_window
        self.priority = priority
        

class DePot:
    def __init__(self, x: float, y: float, id:int):
        self.x = x
        self.y = y
        self.id =id
        


def initialize_vehicles(num_vehicles: int, capacity: int, speed: int) -> List[Vehicle]:
    vehicles = []
    for i in range(num_vehicles):
        vehicles.append(Vehicle(i, capacity, speed))
    return vehicles

def initialize_drop_points(num_drop_points: int,max_distance:float) -> List[DropPoint]:
    drop_points = []
    thetas = [random.uniform(0, 2 * math.pi) for _ in range(num_drop_points)]
    distances = [random.uniform(0, max_distance) for _ in range(num_drop_points)]

    drop_point = [(distances[i]* math.cos(thetas[i]), distances[i] * math.sin(thetas[i])) for i in range(len(thetas))]
    for k in range(num_drop_points):
        x = drop_point[k][0]
        y = drop_point[k][1]
        drop_points.append(DropPoint(x, y, k + 1))
    return drop_points

def initialize_depots(num_depots, radium: float) -> List[DePot]:
    thetas = [random.uniform(0, 2 * math.pi) for _ in range(num_depots)]
    depots = []
    depot = [(radium* math.cos(thetas[i]), radium * math.sin(thetas[i])) for i in range(len(thetas))]
    for k in range(num_depots):
        x = depot[k][0]
        y = depot[k][1]
        depots.append(DePot(x, y, k+1))
    return depots

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# calculate the distance the vehicle needs to travel to deliver the orders
def calculate_route_distance(vehicle: Vehicle, depot: Tuple[float, float]) -> float:
    route_distance = 0
    for i in range(len(vehicle.orders)):
        if i == 0:
            route_distance += calculate_distance(depot, [vehicle.orders[i].destination.x, vehicle.orders[i].destination.y])
        else:
            route_distance += calculate_distance([vehicle.orders[i-1].destination.x, vehicle.orders[i-1].destination.y], [vehicle.orders[i].destination.x, vehicle.orders[i].destination.y])
    route_distance += calculate_distance([vehicle.orders[-1].destination.x, vehicle.orders[-1].destination.y], depot)
    return route_distance