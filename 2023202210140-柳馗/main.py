# main.py

from constants import DRONE_SPEED, TIME_INTERVAL, NODES, EDGES
from graph import initialize_graph, draw_graph, plan_path, choose_center
from order import generate_orders
import time


def main():
    # initialize
    graph = initialize_graph(NODES, EDGES)
    orders = []

    start_time = time.time()  # Start time of simulation


    # Simulate six hours of delivery process
    current_time = 0
    total_distance = 0
    total_flights = 0

    while current_time < 360:  # Six hours in minutes (360 minutes)
        new_orders = generate_orders()
        orders.extend(new_orders)

        # Sort orders by priority (assuming orders are tuples with (order_id, location, priority))
        orders.sort(key=lambda x: x[2], reverse=True)

        # For each priority level, assign orders to delivery centers and plan paths
        for priority in [2, 1, 0]:
            priority_orders = [order for order in orders if order[2] == priority]
            while priority_orders:
                center = choose_center(priority_orders[0], graph)  # Function to choose delivery center
                paths = plan_path(center, priority_orders, graph)  # Function to plan delivery paths
                for path, distance in paths:
                    # Extract details for each delivery
                    order_ids = [order[0] for order in path]
                    path_points = [order[1] for order in path]
                    priorities = [order[2] for order in path]
                    time_taken = (distance / DRONE_SPEED) * 60  # Convert distance to minutes
                    decision = "Immediate" if priorities[0] == 2 else "Batch"
                    
                    # Print detailed delivery information
                    print(
                        f"Current Time: {current_time} minutes, Delivery Center: {center}, Drones Assigned: {len(paths)}, Delivery Path: {path_points}, Destination: {path_points}, Total Distance: {distance} km, Time Taken: {time_taken:.2f} minutes, Priority: {priorities[0]}, Delivery Type: {decision}"
                    )
                    
                    # Update total distance and number of flights
                    total_distance += distance
                    total_flights += 1
                
                # Update the list of orders after deliveries
                priority_orders = [
                    order
                    for order in priority_orders
                    if order not in [o for path in paths for o in path[0]]
                ]

        # Increment simulation time
        current_time += TIME_INTERVAL

    # Simulation end time
    end_time = time.time()

    # Calculate total simulation execution time
    execution_time_ms = (end_time - start_time) * 1000

    # Output total delivery stats over six hours
    total_time_taken = (total_distance / DRONE_SPEED) * 60  # Convert total distance to minutes
    print(
        f"Total Delivery Distance in Six Hours: {total_distance} km, Total Time Taken: {total_time_taken:.2f} minutes, Total Flights: {total_flights}"
    )
    print(f"Total Execution Time: {execution_time_ms:.2f} ms")

    # Additional code for visualization or further processing can be added here
    draw_graph()

if __name__ == "__main__":
    main()
