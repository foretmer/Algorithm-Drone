from typing import List, Tuple
import random
import math
from utils import Vehicle, DropPoint, calculate_distance
from orders_processing import check_path,calculate_path_mileage
import sys

#using the GA to find the shortest path for the vehicle to deliver the orders
def find_shortest_path_GA(vehicle:Vehicle, depot: Tuple[float, float], current_time:float) -> Vehicle:
    #initialize the population
    route_populations = []

    initial_population_num = 5
    generation_num = 5
    order_crossover_rate = 0.8
    mutation_rate = 0.3
    
    original_route = vehicle.route.copy()
    if len(original_route) <5:
        generation_num = 1
    # generate the initial population
    for i in range(initial_population_num):
        route_population = []
        original_route = vehicle.route.copy()
        while (len(original_route)):
            point = random.choice(original_route)
            route_population.append(point)
            original_route.remove(point)
        route_populations.append(route_population)
            
    route_populations.append(vehicle.route)
    
    optimal_route = vehicle.route.copy()
    for generation in range(generation_num):
        #print(f"Generation: {generation}")

        #select the route which satisfy the i time window constraints
        selected_routes = []
        for i in range(len(route_populations)):
            if check_path(vehicle.orders, depot, vehicle, start_time=current_time, route=route_populations[i]):
                selected_routes.append(route_populations[i])
        
        if selected_routes == []:
            #print("No route satisfy the time window constraints.")
            break

        #calculate the fitness of the selected routes
        selected_fitness_distances = []
        for route in selected_routes:
            fitness_distance = calculate_path_mileage(route, depot)
            selected_fitness_distances.append(fitness_distance)
        

        #select the shortest route as the optimal route
        i_optimal_route = selected_routes[selected_fitness_distances.index(min(selected_fitness_distances))]
    
        #update the optimal route
        if min(selected_fitness_distances) < calculate_path_mileage(optimal_route, depot):
            optimal_route = i_optimal_route
            vehicle.route = optimal_route
            vehicle.real_mileage = calculate_path_mileage(optimal_route, depot)


        if generation_num >  1:
            #calculate the probability of each route to be selected as the parent route
            fitness_sum = sum(1/selected_fitness_distance for selected_fitness_distance in selected_fitness_distances)
            probabilities = [fitness_distance/fitness_sum for fitness_distance in selected_fitness_distances]

            #using the roulette wheel selection to select the parent routes
            selected_routes = random.choices(selected_routes, weights=probabilities, k=initial_population_num)
            
            next_generation = selected_routes

            for i in range(len(selected_routes)):
                # generate a probability between 0 and 1, if the probability is less than the order crossover rate, the route will be crossovered
                if random.random() < order_crossover_rate:
                    #select another route to crossover
                    route2 = random.choice(selected_routes)
                    route1 = selected_routes[i]
                    [route1, route2] = order_crossover(route1, route2)
                    next_generation.append(route1)
                    next_generation.append(route2)

                #generate a probability between 0 and 1, if the probability is less than the mutation rate, the route will be mutated
                if random.random() < mutation_rate:
                    mutated_route = mutation(selected_routes[i])
                    next_generation.append(mutated_route)

            route_population = next_generation
    
    #calculate the total distance of the optimal route
    return vehicle

def mutation(selected_route:List[DropPoint]) -> List[DropPoint]:
    num_mutated_points = random.randint(1, len(selected_route))
    # shuffle the mutated points in the selected route
    for i in range(num_mutated_points):
        k = random.randint(0, len(selected_route)-1)
        j = random.randint(0, len(selected_route)-1)
        if k != j:
            selected_route[k], selected_route[j] = selected_route[j], selected_route[k]
    return selected_route

def order_crossover(route1:List[DropPoint], route2:List[DropPoint]) -> Tuple[List[DropPoint], List[DropPoint]]:
    #select two points randomly from the route
    point1 = random.randint(0, len(route1)-1)
    point2 = random.randint(0, len(route1)-1)
    if point1 > point2:
        point1, point2 = point2, point1
    #find crossover part
    crossover_part1 = route1[point1:point2]
    crossover_part2 = route2[point1:point2]

    left_route1 = []
    left_route2 = []

    for i in route1:
        if i not in crossover_part2:
            left_route1.append(i)
    
    for j in route2:
        if j not in crossover_part1:
            left_route2.append(j)
    
    #insert the crossover part2 to the route1
    new_route1 = route1
    new_route1[:point1] = left_route1[:point1]
    new_route1[point2:] = left_route1[point1:]
    new_route1[point1:point2] = crossover_part2

    #insert the crossover part1 to the route2
    new_route2 = route2
    new_route2[:point1] = left_route2[:point1]
    new_route2[point2:] = left_route2[point1:]
    new_route2[point1:point2] = crossover_part1
   
    return [new_route1, new_route2]
    