from typing import List, Tuple
import random
import math
from utils import calculate_distance, Order, Vehicle, DropPoint, initialize_vehicles

# classify the orders by its destination belonging to the sam depot
def classify_orders_by_depot(orders: List[Order]) -> List[Order]:
    orders_by_depot = {}
    for order in orders:
        if order.destination.depot.id in orders_by_depot:
            orders_by_depot[order.destination.depot.id].append(order)
        else:
            orders_by_depot[order.destination.depot.id] = [order]
    return orders_by_depot

def generate_orders(current_orders_num: int, drop_points: List[DropPoint], current_time: int, max_orders: int) -> List[Order]:
    orders = []
    order_id = current_orders_num
    for i in range(len(drop_points)):
        for j in range(random.randint(0, max_orders)):
            order_id += 1
            destination = drop_points[i]
            demand = 1  # Assuming each order has a demand of 1
            priority = random.choice([1, 2, 3])
            if priority == 1:
                time_window = (current_time, current_time + 180)  # 3 hours
            elif priority == 2:
                time_window = (current_time, current_time + 90)   # 1.5 hours
            else:
                time_window = (current_time, current_time + 30)   # 0.5 hour
            #print(order_id, destination,destination.id, demand, time_window, priority)
            orders.append(Order(order_id,destination , demand, time_window, priority))
    return orders

def sort_orders_by_window_end(orders: List[Order]) -> List[Order]:
    return sorted(orders, key=lambda order: order.time_window[1])

def check_orders_due(orders: List[Order], vehicle_speed: int, dealing_window:Tuple[float, float] ) -> List[Order]:
    due_orders = []
    for order in orders:
        depot = order.destination.depot
        depot = [depot.x, depot.y]
        transport_time = calculate_distance(depot, [order.destination.x, order.destination.y]) / vehicle_speed
        if order.time_window[1] -  transport_time <= dealing_window[1]: 
            # print(f"order.time_window:{order.time_window})")
            # print(f"dealing_window:{dealing_window})")
            # print(f"Transport time: {transport_time}")
            due_orders.append(order)
    if len(due_orders) == 0:
        return None
    else:
        # print(len(due_orders))
        # for due_order in due_orders:
        #     print(f"Due orders at drop point {due_order.destination.id}:")
        #     print(f"Order ID: {due_order.order_id}, destination.id: {due_order.destination.id},depot_id:{due_order.destination.depot.id} Demand: {due_order.demand}, Time Window: {due_order.time_window}, Priority: {due_order.priority}")
        return due_orders

# remove the orders that are due from the orders_to_be_delivered list
def remove_due_orders(orders_to_be_delivered: List[Order], due_orders:List[Order]) -> List[Order]:
    for due_order in due_orders:
        orders_to_be_delivered.remove(due_order)
    return orders_to_be_delivered

# allocate the orders which are due to the same destination to the same vehicle
def initial_orders_to_vehicle(orders:List[Order], vehicle_id: int) -> List[Vehicle]:
    allocated_vehicles = []
    vehicle_capacity = 30

    # calculate the number of vehicles needed to deliver the orders
    num_vehicles_needed = math.ceil(sum([order.demand for order in orders]) / vehicle_capacity)
    
    for i in range(num_vehicles_needed):
        allocated_vehicles = initialize_vehicles(num_vehicles_needed, vehicle_capacity, 1)
    
    # allocate the orders to the vehicles
    id = vehicle_id
    i = 0
    while len(orders) != 0:
        while allocated_vehicles[i].load < vehicle_capacity:
            allocated_vehicles[i].vehicle_id = id + i  + 1
            allocated_vehicles[i].orders.append(orders.pop(0))
            allocated_vehicles[i].load += 1
            if allocated_vehicles[i].load == vehicle_capacity:
                i += 1
            if len(orders) == 0:
                # print("All orders are allocated to vehicles.")
                break
    return allocated_vehicles

# initialize the vehicle plan for the orders that are due in the current dealing window, orders's destination are the same can be delivered by the same vehicle
def initial_delivery_plan(due_orders:List[Order] , all_vehicles:List[Vehicle], depot: Tuple[float, float], current_time:float) -> List[Vehicle]:
    # classify the orders by its destination
    orders_by_destination = {}
    for order in due_orders:
        if order.destination.id in orders_by_destination:
            orders_by_destination[order.destination.id].append(order)
        else:
            orders_by_destination[order.destination.id] = [order]
    
    allocated_vehicles = []
    vehicle_id = len(all_vehicles)
    # initialize the route for each vehicle, each vechicle deals with the orders that are due at the same drop point
    for destination_id, orders in orders_by_destination.items():
        if len(orders) != 0:
            allocated_vehicles_i = initial_orders_to_vehicle(orders, vehicle_id = vehicle_id)
            vehicle_id += len(allocated_vehicles_i)
            # print(f"Number of vehicles allocated: {len(allocated_vehicles_i)}")
            allocated_vehicles.extend(allocated_vehicles_i)

    # generate the route for each vehicle
    for vehicle in allocated_vehicles:
        check_path(vehicle.orders, depot, vehicle, current_time, route=None)

    return allocated_vehicles

# aggregate vechicles until there is no more vehicle can be aggregated
def vehicles_aggregate(vehicles: List[Vehicle], depot: Tuple[float, float],current_time:float) -> List[Vehicle]:
    start_id = vehicles[0].vehicle_id 
    aggregatation_flag = True
    while aggregatation_flag:
        aggregatation_flag = False
        for i in range(len(vehicles)):
            for j in range(i+1, len(vehicles)):
                # print(f"aggregate vehicle {vehicles[i].vehicle_id} and vehicle {vehicles[j].vehicle_id}")
                if check_aggregate_capacity(vehicles[i], vehicles[j]) and check_aggregate_time(vehicles[i], vehicles[j], depot, current_time):
                    vehicles[i].orders.extend(vehicles[j].orders)
                    vehicles[i].load += vehicles[j].load
                    vehicles.pop(j)
                    aggregatation_flag = True
                    break
            if aggregatation_flag:
                break
    
    for vehicle in vehicles:
        vehicle.vehicle_id = start_id
        start_id += 1

    return vehicles
                
# check the capacity of the aggregated vehicle
def check_aggregate_capacity(vehicle1: Vehicle, vehicle2: Vehicle) -> bool:
    capacity = vehicle1.capacity
    if vehicle1.load + vehicle2.load <= capacity:
        #print("The aggregated vehicle has enough capacity.")
        return True
    else:
        #print("The aggregated vehicle does not have enough capacity.")
        return False

# check the time is enough for the aggregated vehicle to deliver the orders
def check_aggregate_time(vehicle1: Vehicle, vehicle2: Vehicle, depot: Tuple[float, float], start_time:float) -> bool:
    orders = []
    orders.extend(vehicle1.orders)
    orders.extend(vehicle2.orders)
    if check_path(orders, depot, vehicle1,start_time, route=None):
        #print("The aggregated vehicle can deliver the orders in time.")
        return True
    else:
        #print("The aggregated vehicle cannot deliver the orders in time.")
        return False


def generate_path_orders_from_orders(orders: List[Order], speed: float,route:List[DropPoint] ) -> List[Order]:
    path_orders = []
    speed = speed
        
    #classify the orders by its destination
    orders_by_destination = {}
    for order in orders:
        if order.destination.id in orders_by_destination:
            orders_by_destination[order.destination.id].append(order)
        else:
            orders_by_destination[order.destination.id] = [order]
    
    # find the orders that are due the earliest at each drop point
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


#check the path is available for the vehicle to deliver the orders
def check_path(vehicle_orders: List[Order], depot: Tuple[float, float], vehicle:Vehicle, start_time:float, route:List[DropPoint]) -> bool:
    vehicle_orders = sort_orders_by_window_end(vehicle_orders)
    speed = vehicle.speed
    path_orders = generate_path_orders_from_orders(vehicle_orders, speed, route)
    real_window_ends = []
    # calculate the real window end for each order
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
    
    vehicle_route = []
    for path_order in path_orders:
        vehicle_route.append(path_order.destination)
    real_mileage = calculate_path_mileage(vehicle_route, depot)
    if  real_mileage <= vehicle.max_mileage:
        vehicle.real_mileage = real_mileage
        vehicle.route = vehicle_route
        return True
    else:
        #print("out of the max mileage")
        return False

def calculate_path_mileage(vehicle_path:List[DropPoint], depot: Tuple[float, float]) -> float:
    vehicle_real_mileage = 0
    for i in range(len(vehicle_path)):
        if i == 0:
            vehicle_real_mileage += calculate_distance(depot, [vehicle_path[i].x, vehicle_path[i].y])
        else:
            vehicle_real_mileage += calculate_distance([vehicle_path[i-1].x,vehicle_path[i-1].y], [vehicle_path[i].x, vehicle_path[i].y])
    vehicle_real_mileage += calculate_distance([vehicle_path[-1].x,vehicle_path[-1].y], depot)
    return vehicle_real_mileage