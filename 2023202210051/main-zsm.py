import matplotlib.pyplot as plt
import random
import math
import heapq
from collections import defaultdict

# plt.rc("font", family='Microsoft YaHei')

# 参数设置
j = 3  # 配送中心数量
k = 30  # 卸货点数量
t = 20  # 决策时间段（分钟）
max_capacity = 5  # 无人机最大容量
m = 5  # 每次生成订单的最大数量
max_distance = 20  # 公里
drone_speed = 60 / 60  # 公里/分钟
priority_limits = {'urgent': 0.5 * 60, 'semi-urgent': 1.5 * 60, 'normal': 3 * 60}  # 分钟
max_coord = 10  # 地图最大坐标

order_id = 0  # 全局订单ID

# 手动设置配送中心和卸货点坐标
distribution_centers = [(max_coord / 2, max_coord * 0.8), (max_coord * 0.2, max_coord * 0.2), (max_coord * 0.8, max_coord * 0.2)]
delivery_points = [(random.uniform(0, max_coord), random.uniform(0, max_coord)) for _ in range(k)]

# 绘制配送区域图
plt.figure(figsize=(10, 10))
plt.scatter([p[0] for p in distribution_centers], [p[1] for p in distribution_centers], color='red', label='center')
plt.scatter([p[0] for p in delivery_points], [p[1] for p in delivery_points], color='blue', label='point')
plt.title('map')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig("地图")

# 生成订单
def generate_orders(current_time):
    global order_id
    orders = []
    for point in range(k):
        for _ in range(random.randint(0, m)):
            order = {
                'id': order_id,
                'time': current_time,
                'point': point,
                'priority': random.choice(['urgent', 'semi-urgent', 'normal']),
            }
            orders.append(order)
            order_id += 1
    return orders

# 距离计算
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# 路径规划（改进的最近邻算法）
def nearest_neighbor_algorithm(center, orders):
    routes = []
    while orders:
        route = []
        current_point = center
        current_distance = 0
        current_time = 0
        current_capacity = 0
        
        while orders and current_capacity < max_capacity and current_distance < max_distance / 2:
            next_order = None
            min_distance = float('inf')
            
            for order in orders:
                distance = calculate_distance(current_point, delivery_points[order['point']])
                delivery_time = current_time + distance / drone_speed
                
                if (current_capacity + 1) <= max_capacity and (current_distance + distance) <= max_distance / 2:
                    if distance < min_distance and delivery_time <= priority_limits[order['priority']]:
                        next_order = order
                        min_distance = distance
            
            if next_order is None:
                break
            route.append(next_order)
            orders.remove(next_order)
            current_distance += min_distance
            current_time += min_distance / drone_speed
            current_capacity += 1
            current_point = delivery_points[next_order['point']]
        
        routes.append(route)
    
    return routes

def assign_drone(center, route):
    current_point = center
    current_time = 0
    current_capacity = 0

    for order in route:
        distance_to_order = calculate_distance(current_point, delivery_points[order['point']])
        travel_time = distance_to_order / drone_speed
        current_time += travel_time

        if (current_capacity + 1) > max_capacity:
            print(f"Drone capacity exceeded. Unable to deliver order at point {order['point']}")
            break
        
        if current_time - order['time'] > priority_limits[order['priority']]:
            print(f"Failed to deliver order at point {order['point']} within the required time")
        else:
            print(f"Order at point {order['point']} delivered at time {current_time:.2f} minutes")

        current_point = delivery_points[order['point']]
        current_capacity += 1

    
    distance_to_center = calculate_distance(current_point, center)
    return_time = distance_to_center / drone_speed
    current_time += return_time

    print(f"Drone returned to center at time {current_time:.2f} minutes")

def genetic_algorithm(initial_routes):
    population_size = 50
    generations = 100
    mutation_rate = 0.1

    def calculate_delivery_time(current_position, order):
        distance = calculate_distance(current_position, delivery_points[order['point']])
        travel_time = distance / drone_speed
        delivery_time = current_time + travel_time
        
        if order['priority'] == 'normal':
            time_limit = 180
        elif order['priority'] == 'semi-urgent':
            time_limit = 90
        elif order['priority'] == 'urgent':
            time_limit = 30
        
        return delivery_time

    def initial_population():
        population = []
        for _ in range(population_size):
            route = initial_routes[:]
            random.shuffle(route)
            population.append(route)
        return population

    def fitness(route):
        total_distance = 0
        penalty = 0

        for i in range(len(route) - 1):
            total_distance += calculate_distance(delivery_points[route[i]['point']], delivery_points[route[i + 1]['point']])

        for order in route:
            delivery_time = calculate_delivery_time(delivery_points[0], order)
            if delivery_time > priority_limits[order['priority']]:
                penalty += 1000

        fitness_value = total_distance - penalty

        return fitness_value

    def crossover(parent1, parent2):
        if len(parent1) < 3 or len(parent2) < 3:
            # 如果任一路线长度小于3，不进行交叉或以特殊方式处理
            return parent1[:], parent2[:]

        crossover_point1 = random.randint(1, len(parent1) - 2)
        crossover_point2 = random.randint(1, len(parent2) - 2)

        # 确保交叉点不同
        if crossover_point1 == crossover_point2:
            crossover_point2 = (crossover_point2 + 1) % len(parent2)

        child1 = parent1[:crossover_point1] + [gene for gene in parent2 if  gene not in parent1[:crossover_point1]]
        child2 = parent2[:crossover_point2] + [gene for gene in parent1 if  gene not in parent2[:crossover_point2]]

        return child1, child2

    def mutate(route):
        for i in range(len(route)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(route) - 1)
                route[i], route[j] = route[j], route[i]
        return route

    def select_parents(population):
        fitnesses = [fitness(route) for route in population]
        total_fitness = sum(fitnesses)
        if total_fitness == 0:  # 添加检查以避免除以零
            return population[0], population[1]  # 或者选择其他适当的处理方式
        probabilities = [fit / total_fitness for fit in fitnesses]
        parents = []
        for _ in range(2):
            r = random.random()
            cumulative_probability = 0
            for i, probability in enumerate(probabilities):
                cumulative_probability += probability
                if r < cumulative_probability:
                    parents.append(population[i])
                    break
        return parents

    population = initial_population()

    for _ in range(generations):
        new_population = []

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = sorted(new_population, key=lambda route: fitness(route))[:population_size]

    best_route = min(population, key=lambda route: fitness(route))
    return best_route

def plot_routes_all_centers(center_routes_dict, current_time):
    plt.figure(figsize=(10, 10))
    plt.scatter([p[0] for p in distribution_centers], [p[1] for p in distribution_centers], color='red', label='center')
    plt.scatter([p[0] for p in delivery_points], [p[1] for p in delivery_points], color='blue', label='point')
    
    colors = ['green', 'purple', 'orange']
    
    for center_idx, (center, routes) in enumerate(center_routes_dict.items()):
        for idx, route in enumerate(routes):
            route_points = [center] + [delivery_points[order['point']] for order in route] + [center]
            plt.plot([p[0] for p in route_points], [p[1] for p in route_points], color=colors[center_idx % len(colors)], label=f'drone_path {center_idx + 1}-{idx + 1}')
    
    plt.title(f'drone path at time {current_time} minutes')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(f"无人机路径_{current_time}_minutes")
    plt.close()

def calculate_total_distance(center, route):
    total_distance = 0
    current_point = center
    for order in route:
        total_distance += calculate_distance(current_point, delivery_points[order['point']])
        current_point = delivery_points[order['point']]
    total_distance += calculate_distance(current_point, center)  # Return to center
    return total_distance

# 主决策循环
orders_dict = {}
orders_heap = []
cumulative_total_distance = 0

for current_time in range(0, 24 * 60, t):
    new_orders = generate_orders(current_time)
    for order in new_orders:
        orders_dict[order['id']] = order
        priority = 0 if order['priority'] == 'urgent' else 1 if order['priority'] == 'semi-urgent' else 2
        priority = 3 - priority
        heapq.heappush(orders_heap, (priority, order['time'], order['id']))
    # 打印当前时间和所有待处理订单
    print(f"Time: {current_time} minutes")
    print("Pending orders:")
    for order_id, order in orders_dict.items():
        print(f"Order ID: {order_id}, Point: {order['point']}, Priority: {order['priority']}, Time: {order['time']}")

    pending_orders = []
    while orders_heap:
        priority, order_time, order_id = heapq.heappop(orders_heap)
        if current_time - order_time <= priority_limits[orders_dict[order_id]['priority']]:
            order = orders_dict[order_id]
            pending_orders.append(order)
        else:
            print(f"Order at point {order['point']} expired and was removed from the queue")

    center_orders = defaultdict(list)
    for order in pending_orders:
        closest_center = min(distribution_centers, key=lambda center: calculate_distance(center, delivery_points[order['point']]))
        center_orders[closest_center].append(order)

    center_routes_dict = defaultdict(list)
    current_total_distance = 0

    for center_idx, (center, orders) in enumerate(center_orders.items()):
        initial_routes = nearest_neighbor_algorithm(center, orders)
        optimized_routes = [genetic_algorithm(route) for route in initial_routes]
        center_routes_dict[center] = optimized_routes

        for route in optimized_routes:
            assign_drone(center, route)
            for order in route:
                pending_orders.remove(order)
            
            current_total_distance += calculate_total_distance(center, route)

    cumulative_total_distance += current_total_distance

    print(f"Current total distance at time {current_time}: {current_total_distance:.2f} km")
    print(f"Cumulative total distance at time {current_time}: {cumulative_total_distance:.2f} km")

    plot_routes_all_centers(center_routes_dict, current_time)

    for order in pending_orders:
        priority = 0 if order['priority'] == 'urgent' else 1 if order['priority'] == 'semi-urgent' else 2
        if order in pending_orders:
            heapq.heappush(orders_heap, (priority, order['time'], order['id'], order))
        else:
            print(f"Order {order['id']} not found in all_orders. Skipping heappush.")
