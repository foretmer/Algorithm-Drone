import richuru

richuru.install()

from utils import *
from omegaconf import OmegaConf
from loguru import logger
import argparse


def gene_algo(drone: Drone, depot: Tuple[float, float]):
    route_population = []

    initial_population_num = 100
    generation_num = 100
    for i in range(initial_population_num):
        route = drone.route.copy()
        random.shuffle(route)
        if route not in route_population:
            route_population.append(route)

    if len(route_population) <= initial_population_num / 2:
        generation_num = 1

    for _ in range(generation_num):

        selected_routes = []
        for i in range(len(route_population)):
            if check_path(drone.orders, depot, drone, start_time=0, route=route_population[i]):
                selected_routes.append(route_population[i])

        selected_fitness_distances = []
        for route in selected_routes:
            fitness_distance = 0
            for i in range(len(route)):
                if i == 0:
                    fitness_distance += calculate_distance(depot, [route[i].x, route[i].y])
                else:
                    fitness_distance += calculate_distance([route[i - 1].x, route[i - 1].y], [route[i].x, route[i].y])
            fitness_distance += calculate_distance([route[-1].x, route[-1].y], depot)
            
            if fitness_distance <= 20:
                selected_fitness_distances.append(fitness_distance)


        optimal_route = selected_routes[0]

        zip_selected_routes = zip(selected_routes, selected_fitness_distances)
        for route, fitness_distance in sorted(zip_selected_routes, key=lambda x: x[1]):
            selected_routes.append(route)
        selected_routes = selected_routes[: math.ceil(len(selected_routes) / 2)]

        next_generation = selected_routes

        for i in range(len(selected_routes)):
            mutation_rate = 0.1
            
            if random.random() < mutation_rate:
                mutated_route = mutation(selected_routes[i])
                if mutated_route not in next_generation:
                    next_generation.append(mutated_route)

        route_population = next_generation
        if len(route_population) == 1:
            break

    drone.route = optimal_route


def mutation(selected_route: list[DropPoint]) -> list[DropPoint]:
    num_mutated_points = random.randint(1, len(selected_route))
    for _ in range(num_mutated_points):
        k = random.randint(0, len(selected_route) - 1)
        j = random.randint(0, len(selected_route) - 1)
        if k != j:
            selected_route[k], selected_route[j] = selected_route[j], selected_route[k]
    return selected_route


def main(args):
    if args.config is None:
        args.config = "configs/base.yaml"

    conf = OmegaConf.load(args.config)

    drop_points_num = conf.drop_points_num
    depots_num = conf.depots_num
    drone_speed = conf.drone_speed
    max_distance = conf.max_distance
    depot = (0, 0)

    one_round_order_time = conf.one_round_order_time
    simulation_time = conf.simulation_time
    max_order_num = conf.max_order_num
  
    drop_points = initial_drop_points(drop_points_num, max_distance)

    depots = initial_depots(depots_num, max_distance)

    current_time = 0
    all_drones_distance = 0

    orders_to_be_delivered = []

    all_drones = []
    current_orders_num = 0
    round_num = 0

    distribute_depot(drop_points, depots)

    while current_time <= simulation_time:
        if current_time % one_round_order_time == 0:
            logger.info("New Round")
            new_orders = generate_new_orders(current_orders_num, drop_points, current_time, max_order_num)
            current_orders_num += len(new_orders)
            orders_to_be_delivered.extend(new_orders)
            orders_to_be_delivered = sort_orders_by_window_end(orders_to_be_delivered)
            dealing_window = [current_time, current_time + one_round_order_time]

            due_orders = check_time_due(orders_to_be_delivered, drone_speed=drone_speed,
                                          dealing_window=dealing_window)

            if due_orders:
                due_orders_by_depots = classify_orders_by_depot(due_orders)
                for depot_id, due_orders_by_depot in due_orders_by_depots.items():
                    logger.info("allocate due orders to depot...")
                    logger.info(f"start from depot {depot_id}...")
                    depot = depots[depot_id - 1]
                    depot = [depot.x, depot.y]
                    
                    allocated_drones = allocate_drone_num(due_orders_by_depot, all_drones, depot, current_time)

                    drones_path = drones_merge(allocated_drones, depot, current_time)
                    #drones_path = allocated_drones

                    for drone in drones_path:
                        drone.start_depot = depot_id
                        logger.info(f"allocate path for drone {drone.drone_id}")
                        gene_algo(drone, depot)

                    all_drones.extend(drones_path)

                    orders_to_be_delivered = remove_due_orders(orders_to_be_delivered, due_orders_by_depot)
                    
                    if drones_path:
                        all_drones_distance += Draw(drop_points, depots, drones_path, round_num).draw_route()

        round_num += 1
        current_time += 1

    logger.info(f"The total distance covered by all drones is {all_drones_distance} km")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="yaml file for config.")
    args = parser.parse_args()
    main(args)
