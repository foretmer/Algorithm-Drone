import richuru
richuru.install()

import math
import random
import numpy as np
from loguru import logger

from roles import *

#注释：初始化车辆，每个车辆的载重量为capacity，速度为speed，每个车辆的初始位置为depot   
def initial_drones(num_drones: int, capacity: int, speed: int) -> list[Drone]:
    drones = []
    for i in range(num_drones):
        drones.append(Drone(i, capacity, speed))
    return drones

def generate_spaced_distances(num_points: int, max_distance: float, reverse=False) -> list[float]:
    uniform_points = np.linspace(0, max_distance, num_points)
    
    jitter = np.random.uniform(-0.5, 0.5, num_points) * (max_distance / num_points)
    jittered_points = uniform_points + jitter
    
    jittered_points = np.clip(jittered_points, 0, max_distance)
    
    result = jittered_points.tolist()
    
    if reverse:
        return result[::-1]
    else:
        return result

def initial_drop_points(num_drop_points: int, max_distance:float) -> list[DropPoint]:
    drop_points = []
    angles = generate_spaced_distances(num_drop_points, 2 * math.pi)
    distances = generate_spaced_distances(num_drop_points, max_distance)

    drop_point = [(distances[i]* math.cos(angles[i]), distances[i] * math.sin(angles[i])) for i in range(len(angles))]
    for k in range(num_drop_points):
        x = drop_point[k][0]
        y = drop_point[k][1]
        drop_points.append(DropPoint(x, y, k + 1))
    return drop_points

def initial_depots(num_depots: int, max_distance:float) -> list[DePot]:
    depots = []
    angles = generate_spaced_distances(num_depots, 2 * math.pi, True)
    distances = generate_spaced_distances(num_depots, max_distance, True)
    
    depot = [(distances[i]* math.sin(angles[i]), distances[i] * math.cos(angles[i])) for i in range(len(angles))]
    for k in range(num_depots):
        x = depot[k][0]
        y = depot[k][1]
        depots.append(DePot(x, y, k + 1))
    return depots

#多车厂转为单车厂
def distribute_depot(drop_points: list[DropPoint], depots: list[DePot]):
    for drop_point in drop_points:
        min_distance = float('inf')
        for depot in depots:
            distance = calculate_distance([drop_point.x, drop_point.y], [depot.x, depot.y])
            if distance < min_distance:
                min_distance = distance
                drop_point.depot = depot

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def generate_new_orders(current_orders_num: int, drop_points: list[DropPoint], current_time: int, max_orders: int) -> list[Order]:
    orders = []
    order_id = current_orders_num
    for i in range(len(drop_points)):
        for _ in range(random.randint(0, max_orders)):
            order_id += 1
            destination = drop_points[i]
            demand = 1  
            priority = random.choice([0, 1, 2])
            if priority == 0:
                time_window = (current_time, current_time + 180)  
                threshold = 180
                
            elif priority == 1:
                time_window = (current_time, current_time + 90)   
                threshold = 90
            
            else:
                time_window = (current_time, current_time + 30) 
                threshold = 30
            
            if calculate_distance([destination.x, destination.y], [drop_points[i].x, drop_points[i].y]) / 1 <= threshold:
                orders.append(Order(order_id,destination , demand, time_window, priority))
    
    return orders

def classify_orders_by_depot(orders: list[Order]) -> list[Order]:
    orders_by_depot = {}
    for order in orders:
        if order.destination.depot.id in orders_by_depot:
            orders_by_depot[order.destination.depot.id].append(order)
        else:
            orders_by_depot[order.destination.depot.id] = [order]
    return orders_by_depot

def sort_orders_by_window_end(orders: list[Order]) -> list[Order]:
    return sorted(orders, key=lambda order: order.time_window[1])

def check_time_due(orders: list[Order], drone_speed: int, dealing_window:Tuple[float, float] ) -> list[Order]:
    due_orders = []
    for order in orders:
        depot = order.destination.depot
        depot = [depot.x, depot.y]
        transport_time = calculate_distance(depot, [order.destination.x, order.destination.y]) / drone_speed
        if order.time_window[1] -  transport_time <= dealing_window[1]: 
            due_orders.append(order)
    if len(due_orders) == 0:
        return None
    else:
        logger.info("Due orders delivering...")
        logger.info(f"Due orders: {len(due_orders)}")
        return due_orders

def remove_due_orders(orders_to_be_delivered: list[Order], due_orders:list[Order]) -> list[Order]:
    for due_order in due_orders:
        orders_to_be_delivered.remove(due_order)
    return orders_to_be_delivered

def greedy_drone_num(orders:list[Order], drone_id: int) -> list[Drone]:
    allocated_drones = []
    drone_capacity = 25
    
    num_drones_needed = math.ceil(sum([order.demand for order in orders]) / drone_capacity)
    for i in range(num_drones_needed):
        allocated_drones = initial_drones(num_drones_needed, drone_capacity, 1)
    
    id = drone_id
    i = 0
    while len(orders) != 0:
        while allocated_drones[i].load < drone_capacity:
            allocated_drones[i].drone_id = id + i  + 1
            allocated_drones[i].orders.append(orders.pop(0))
            allocated_drones[i].load += 1
            if allocated_drones[i].load == drone_capacity:
                i += 1
            if len(orders) == 0:
                break
    return allocated_drones

def allocate_drone_num(due_orders:list[Order] , all_drones:list[Drone], depot: Tuple[float, float], current_time:float) -> list[Drone]:

    orders_by_destination = {}
    for order in due_orders:
        if order.destination.id in orders_by_destination:
            orders_by_destination[order.destination.id].append(order)
        else:
            orders_by_destination[order.destination.id] = [order]
    
    allocated_drones = []
    drone_id = len(all_drones)
    for _, orders in orders_by_destination.items():
        if len(orders) != 0:
            allocated_drones_i = greedy_drone_num(orders, drone_id = drone_id)
            drone_id += len(allocated_drones_i)
            allocated_drones.extend(allocated_drones_i)

    for drone in allocated_drones:
        check_path(drone.orders, depot, drone, current_time, route=None)

    return allocated_drones

def calculate_route_distance(drone: Drone, depot: Tuple[float, float]) -> float:
    route_distance = 0
    for i in range(len(drone.orders)):
        if i == 0:
            route_distance += calculate_distance(depot, [drone.orders[i].destination.x, drone.orders[i].destination.y])
        else:
            route_distance += calculate_distance([drone.orders[i-1].destination.x, drone.orders[i-1].destination.y], [drone.orders[i].destination.x, drone.orders[i].destination.y])
    route_distance += calculate_distance([drone.orders[-1].destination.x, drone.orders[-1].destination.y], depot)
    return route_distance

def drones_merge(drones: list[Drone], depot: Tuple[float, float],current_time:float) -> list[Drone]:
    start_id = drones[0].drone_id 
    aggregatation_flag = True
    while aggregatation_flag:
        aggregatation_flag = False
        for i in range(len(drones)):
            for j in range(i+1, len(drones)):
                if check_drone_capacity(drones[i], drones[j]) and check_drone_time(drones[i], drones[j], depot, current_time):
                    drones[i].orders.extend(drones[j].orders)
                    drones[i].load += drones[j].load
                    drones.pop(j)
                    aggregatation_flag = True
                    break
            if aggregatation_flag:
                break
    
    for drone in drones:
        drone.drone_id = start_id
        start_id += 1

    return drones
                
def check_drone_capacity(drone1: Drone, drone2: Drone) -> bool:
    capacity = drone1.capacity
    if drone1.load + drone2.load <= capacity:
        return True
    else:
        return False

def check_drone_time(drone1: Drone, drone2: Drone, depot: Tuple[float, float], start_time:float) -> bool:
    orders = []
    orders.extend(drone1.orders)
    orders.extend(drone2.orders)
    if check_path(orders, depot, drone1,start_time, route=None):
        return True
    else:
        return False

def generate_path_orders_from_orders(orders: list[Order], speed: float,route:list[DropPoint] ) -> list[Order]:
    path_orders = []
    speed = speed
        
    orders_by_destination = {}
    for order in orders:
        if order.destination.id in orders_by_destination:
            orders_by_destination[order.destination.id].append(order)
        else:
            orders_by_destination[order.destination.id] = [order]
    
    for destination_id, orders in orders_by_destination.items():
        orders = sort_orders_by_window_end(orders)
        path_orders.append(orders[0])
    
    sorted_path_orders = []
    if route is not None:
        for i in range(len(route)):
            destination_id = route[i].id
            sorted_path_order = [order for order in path_orders if order.destination.id == destination_id]
            sorted_path_orders.extend(sorted_path_order)
    else:
        sorted_path_orders = path_orders
    
    return sorted_path_orders


def check_path(drone_orders: list[Order], depot: Tuple[float, float], drone:Drone, start_time:float, route:list[DropPoint]) -> bool:
    drone_orders = sort_orders_by_window_end(drone_orders)
    speed = drone.speed
    path_orders = generate_path_orders_from_orders(drone_orders, speed, route)
    real_window_ends = []
    current_time = start_time
    for i in range(len(path_orders)):
        if i == 0:
            transport_time = calculate_distance(depot, [path_orders[i].destination.x, path_orders[i].destination.y]) / speed
            real_window_end = current_time + transport_time
            real_window_ends.append(real_window_end)
            current_time = real_window_end
        else:
            transport_time = calculate_distance([path_orders[i-1].destination.x, path_orders[i-1].destination.y], [path_orders[i].destination.x, path_orders[i].destination.y]) / speed
            real_window_end = current_time + transport_time
            real_window_ends.append(real_window_end)
            current_time = real_window_end

    for i in range(len(path_orders)):
        if real_window_ends[i] > path_orders[i].time_window[1]:
            return False
        
    drone_route = []
    for path_order in path_orders:
        drone_route.append(path_order.destination)
    drone.route = drone_route
    return True
