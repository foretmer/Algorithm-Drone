import random
import math
from typing import List, Tuple
import json
from utils import initialize_drop_points, initialize_depots
from drop_point_classify import find_closest_depot
from GA_shortest_path import find_shortest_path_GA
from orders_processing import generate_orders, sort_orders_by_window_end, check_orders_due, classify_orders_by_depot, initial_delivery_plan, vehicles_aggregate, remove_due_orders

#simulate the whole process
def simulator(k:int)-> float:
    num_drop_points = 40 # number of drop points
    num_depots = 5 # number of depots
    vehicle_speed = 1  # 1 unit distance per minute
    max_distance = 5 # max distance of the map a circle with radius 5
    
    time_interval = 12  # interval of generating orders
    time_sensitivity = 1 # minute unit in the simulation
    simulation_duration = 8 * 60  # simulation duration in minutes
    max_orders = 5  # Max orders generated per generating interval per drop point

    #generate drop points which distances from depot limited to max_distance randomly
    drop_points = initialize_drop_points(num_drop_points,max_distance)
    
    #generate depots which distances from the origin limited to max_distance randomly
    depots = initialize_depots(num_depots, max_distance)

    current_time = 0

    orders_to_be_delivered = []

    all_vehicles = []
    current_orders_num = 0
    dealing_window = [0,0]

    #find the closest depot for each drop point
    find_closest_depot(drop_points, depots, therehold=0.5)

    while current_time < simulation_duration:
        if current_time % time_interval == 0:
            new_orders = generate_orders(current_orders_num,drop_points, current_time, max_orders)
            current_orders_num += len(new_orders)
            orders_to_be_delivered.extend(new_orders)
            orders_to_be_delivered = sort_orders_by_window_end(orders_to_be_delivered)

            # set the dealing window to be the next k time intervals
            dealing_window = [current_time, current_time + k * time_interval]
           
            #check orders due in the current dealing  window
            due_orders = check_orders_due(orders_to_be_delivered, vehicle_speed=vehicle_speed, dealing_window=dealing_window)

            if not due_orders:
                None
                # print("No orders due in the current dealing window.")
            else:
                due_orders_by_depots = classify_orders_by_depot(due_orders)
                for depot_id, due_orders_by_depot in due_orders_by_depots.items():
                    # print(f'dealing the orders starting from depot {depot_id}...')
                    depot = depots[depot_id-1]
                    depot = [depot.x, depot.y]
                    #initialize the vehicle plan for the orders that are due in the current dealing window, orders's destination are the same can be delivered by the same vehicle
                    allocated_vehicles = initial_delivery_plan(due_orders_by_depot, all_vehicles, depot, current_time)

                    #aggregate vechicles until there is no more vehicle can be aggregated
                    aggregated_vehicles = vehicles_aggregate(allocated_vehicles, depot, current_time)

                    for aggregated_vehicle in aggregated_vehicles:
                        
                        # if check_path(aggregated_vehicle.orders, depot, aggregated_vehicle, start_time=current_time, route=aggregated_vehicle.route) == False:
                        #     print("error: the aggregated vehicle's route is not valid")
                        #     sys.exit(1)
                        
                        # print(f"real mileage of vehicle {aggregated_vehicle.vehicle_id}: {aggregated_vehicle.real_mileage}")
                        # #using the GA to find the shortest path for the vehicle to deliver the orders
                        # print(f"searching for the shortest path for vehicle {aggregated_vehicle.vehicle_id}")
                        aggregated_vehicle = find_shortest_path_GA(aggregated_vehicle, depot, current_time= current_time)


                    #print the aggregated vehicles' orders and routes
                    # for aggregated_vehicle in aggregated_vehicles:
                    #     print(f"Aggregated vehicle {aggregated_vehicle.vehicle_id} has orders:")
                    #     for order in aggregated_vehicle.orders:
                    #         print(f"Order ID: {order.order_id}, destination.id: {order.destination.id}, Demand: {order.demand}, Time Window: {order.time_window}, Priority: {order.priority}")
                    #     print(f"Aggregated vehicle {aggregated_vehicle.vehicle_id} has route:")
                    #     for drop_point in aggregated_vehicle.route:
                    #         print(f"Drop point ID: {drop_point.id}")

                    all_vehicles.extend(aggregated_vehicles)

                    #remove the orders that are due from the orders_to_be_delivered list
                    orders_to_be_delivered = remove_due_orders(orders_to_be_delivered, due_orders_by_depot)
            
        current_time += time_sensitivity
    
    #write all the vehicles' orders and routes to a json file
    whole_vehicle_distance = 0
    vehicles = []
    error_count = 0
    for vehicle in all_vehicles:
        vehicle_dict = {}
        vehicle_dict['vehicle_id'] = vehicle.vehicle_id
        vehicle_dict['orders'] = []
        for order in vehicle.orders:
            order_dict = {}
            order_dict['order_id'] = order.order_id
            order_dict['destination_id'] = order.destination.id
            order_dict['demand'] = order.demand
            order_dict['time_window'] = order.time_window
            order_dict['priority'] = order.priority
            vehicle_dict['orders'].append(order_dict)
        vehicle_dict['route'] = [drop_point.id for drop_point in vehicle.route]
        vehicle_dict['real_mileage'] = vehicle.real_mileage
        if vehicle.real_mileage > vehicle.max_mileage:
            error_count += 1
        whole_vehicle_distance += vehicle.real_mileage
        vehicles.append(vehicle_dict)

    with open('vehicles.json', 'w') as f:
        json.dump(vehicles, f, indent=4)
    
    print(f"Whole vehicles' distance: {whole_vehicle_distance}")
    return whole_vehicle_distance


