import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from deap import base, creator, tools, algorithms

# 配送中心位置和名称
delivery_center_locations = np.array([[44, 47], [0, 3], [3, 39]])
delivery_center_names = ['Center 1', 'Center 2', 'Center 3']

# 订单位置和优先级
orders = [((9, 19), 'Very Urgent'), ((21, 36), 'Very Urgent'), ((21, 36), 'Urgent'), 
          ((9, 19), 'Urgent'), ((9, 19), 'Urgent'), ((9, 19), 'Urgent'), ((24, 24), 'Urgent'),
          ((9, 19), 'General'), ((21, 36), 'General'), ((23, 6), 'General'), ((37, 48), 'Urgent'), 
          ((36, 37), 'General'), ((39, 43), 'Very Urgent'), ((51, 52), 'General'), ((44, 38), 'Urgent')]

# DBSCAN聚类算法进行配送中心和订单的聚类
def cluster_orders(delivery_center_locations, orders, eps=20):
    print("\n正在进行路径规划...")
    order_locations = np.array([order[0] for order in orders])
    order_priorities = np.array([order[1] for order in orders])
    order_features = np.column_stack((order_locations, order_priorities))
    dbscan = DBSCAN(eps=eps, min_samples=1)
    order_labels = dbscan.fit_predict(order_features)  # 在聚类时考虑优先级信息
    center_orders = {i: [] for i in range(len(delivery_center_locations))}

    for label in set(order_labels):
        cluster_orders = order_features[order_labels == label]
        center_distances = cdist(cluster_orders[:, :-1], delivery_center_locations)  # 只考虑位置信息计算距离
        closest_centers = np.argmin(center_distances, axis=1)

        for order_idx, center_idx in enumerate(closest_centers):
            center_orders[center_idx].append(orders[np.where(order_labels == label)[0][order_idx]])

    return center_orders

# 绘制配送中心位置
plt.figure(figsize=(10, 8))
plt.scatter(delivery_center_locations[:, 0], delivery_center_locations[:, 1], c='blue', marker='s', label='Delivery Centers')
for i, (x, y) in enumerate(delivery_center_locations):
    plt.text(x, y+0.5, f'{delivery_center_names[i]} ({x}, {y})', fontsize=12, ha='center', va='bottom')
plt.grid()
plt.show()

def total_distance(individual):
    # 初始化无人机的配送任务
    num_drones = len(delivery_center_locations)
    drones = [[] for _ in range(num_drones)]
    # 分配订单给无人机
    for order_idx, center_idx in enumerate(individual):
        drones[center_idx].append(order_idx)
    total_distance = 0
    for drone_idx, drone_tasks in enumerate(drones):
        if drone_tasks:
            current_location = delivery_center_locations[drone_idx] # 无人机从配送中心出发
            total_drone_distance = 0
            sorted_tasks = sorted(drone_tasks, key=lambda x: {'Very Urgent': 0, 'Urgent': 1, 'General': 2}[orders[x][1]])
            batches = [sorted_tasks[i:i + 2] for i in range(0, len(sorted_tasks), 2)] # 任务分批，每次最多2个订单
            for batch in batches:
                for order_idx in batch:
                    order_location = orders[order_idx][0]
                    order_priority = orders[order_idx][1]
                    distance_to_order = np.linalg.norm(current_location - np.array(order_location))
                    total_drone_distance += distance_to_order
                    current_location = order_location
                    # 根据订单的紧急程度调整飞行距离的惩罚
                    if order_priority == 'Very Urgent':
                        penalty_factor = 3  # 非常紧急订单的惩罚系数
                    elif order_priority == 'Urgent':
                        penalty_factor = 2  # 紧急订单的惩罚系数
                    else: # order_priority == 'General'
                        penalty_factor = 1  # 普通订单的惩罚系数
                    if distance_to_order > 20:
                        penalty = (distance_to_order - 20) * penalty_factor  # 超出部分按惩罚系数进行惩罚
                        total_drone_distance += penalty
                distance_back_to_center = np.linalg.norm(current_location - delivery_center_locations[drone_idx]) # 无人机回到配送中心
                total_drone_distance += distance_back_to_center
                current_location = delivery_center_locations[drone_idx] # 返回配送中心后更新当前位置为配送中心
            total_distance += total_drone_distance
    return total_distance,

# 配置DEAP工具箱
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("individual", tools.initRepeat, creator.Individual, lambda: random.randint(0, len(delivery_center_locations) - 1), n=len(orders))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", total_distance)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(delivery_center_locations) - 1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 遗传算法
def genetic_algorithm():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)
    best_individual = hof[0]
    return best_individual

# 蚁群优化算法
def ant_colony_optimization():
    # 实现ACO算法
    num_orders = len(orders)
    num_ants = 10
    num_iterations = 100
    evaporation_rate = 0.5
    pheromone = np.ones((num_orders, num_orders))  # 信息素矩阵
    best_individual = [0] * num_orders
    best_distance = float('inf')
    for _ in range(num_iterations):
        all_routes = []
        all_distances = []
        for _ in range(num_ants):
            route = list(np.random.permutation(num_orders))
            distance = sum([np.linalg.norm(np.array(orders[route[i]][0]) - np.array(orders[route[(i + 1) % num_orders]][0])) for i in range(num_orders)])
            all_routes.append(route)
            all_distances.append(distance)
            if distance < best_distance:
                best_distance = distance
                best_individual = route
        # 更新信息素
        pheromone *= (1 - evaporation_rate)
        for route, distance in zip(all_routes, all_distances):
            for i in range(num_orders):
                pheromone[route[i]][route[(i + 1) % num_orders]] += 1.0 / distance
    # 保证返回的个体在0到2之间
    best_individual = [i % len(delivery_center_locations) for i in best_individual]
    return best_individual

# 粒子群优化算法
def particle_swarm_optimization():
    # 实现PSO算法
    num_particles = 30
    num_iterations = 100
    inertia = 0.5
    cognitive = 1.0
    social = 2.0
    best_individual = [0] * len(orders)
    best_distance = float('inf')

    particles = [list(np.random.permutation(len(orders))) for _ in range(num_particles)]
    velocities = [np.random.rand(len(orders)) for _ in range(num_particles)]
    personal_best_positions = particles[:]
    personal_best_distances = [float('inf')] * num_particles

    for _ in range(num_iterations):
        for i, particle in enumerate(particles):
            distance = sum([np.linalg.norm(np.array(orders[particle[j]][0]) - np.array(orders[particle[(j + 1) % len(orders)]][0])) for j in range(len(orders))])
            if distance < personal_best_distances[i]:
                personal_best_distances[i] = distance
                personal_best_positions[i] = particle[:]
            if distance < best_distance:
                best_distance = distance
                best_individual = particle[:]
        
        for i, particle in enumerate(particles):
            for j in range(len(orders)):
                velocities[i][j] = (inertia * velocities[i][j] +
                                    cognitive * random.random() * (personal_best_positions[i][j] - particle[j]) +
                                    social * random.random() * (best_individual[j] - particle[j]))
                particle[j] = int(particle[j] + velocities[i][j]) % len(orders)

    # 保证返回的个体在0到2之间
    best_individual = [i % len(delivery_center_locations) for i in best_individual]
    return best_individual

# 遗传算法 + 蚁群优化 + 粒子群优化混合算法
def hybrid_algorithm():
    # 执行遗传算法
    best_individual_genetic = genetic_algorithm()
    # 执行蚁群优化算法
    best_individual_ant_colony = ant_colony_optimization()
    # 执行粒子群优化算法
    best_individual_particle_swarm = particle_swarm_optimization()
    # 综合三种算法的结果
    # 这里简单地选择总距离最短的个体作为最终结果
    all_individuals = [best_individual_genetic, best_individual_ant_colony, best_individual_particle_swarm]
    best_individual = min(all_individuals, key=lambda x: total_distance(x))
    return best_individual

# 执行混合算法
best_individual = hybrid_algorithm()

# 生成配送路径
best_routes = [[] for _ in range(len(delivery_center_locations))]
added_departure = [False] * len(delivery_center_locations)
assigned_orders = set()  # 用于跟踪已分配的订单
for idx, center_idx in enumerate(best_individual):
    if idx not in assigned_orders:  # 如果订单未被分配给其他无人机
        if not added_departure[center_idx]:
            best_routes[center_idx].append((tuple(delivery_center_locations[center_idx]), 'Depart from Center'))
            added_departure[center_idx] = True
        best_routes[center_idx].append(orders[idx])
        assigned_orders.add(idx)  # 将订单标记为已分配

# 定义分批次配送函数
def create_batches(routes):
    batched_routes = []
    for route in routes:
        batched_route = [route[0]]
        batch = []
        for stop in route[1:]:
            if stop[1] != 'Depart from Center':  # 如果不是出发点
                batch.append(stop)
            if len(batch) == 2:
                batched_route.extend(batch)
                batched_routes.append(batched_route)
                batched_routes[-1].append((batched_route[0][0], 'Return to Center'))
                batched_route = [route[0]]  # 重新开始一个新批次
                batch = []
        if batch:  # 处理剩余的订单
            batched_route.extend(batch)
            batched_routes.append(batched_route)
            batched_routes[-1].append((batched_route[0][0], 'Return to Center'))
    return batched_routes

# 调用分批次配送函数
batched_routes = create_batches(best_routes)

print("\n最终的配送路径:")
for i, route in enumerate(batched_routes):
    print(f"无人机 {i + 1} 的路径:")
    print(" -> ".join(map(str, route)))

# 定义绘制配送路径函数
def plot_routes(delivery_center_locations, orders, batched_routes, delivery_center_names):
    plt.figure(figsize=(10, 8))
    plt.scatter(delivery_center_locations[:, 0], delivery_center_locations[:, 1], c='blue', marker='s', label='Delivery Centers')
    for i, (x, y) in enumerate(delivery_center_locations):
        plt.text(x, y + 0.8, f'{delivery_center_names[i]} ({x}, {y})', fontsize=12, ha='center', va='bottom')
    for order in orders:
        order_x, order_y = order[0]
        plt.scatter(order_x, order_y, c='red', marker='o')
        plt.text(order_x, order_y + 1, f'({order_x}, {order_y})', fontsize=10, ha='center', va='bottom')

    for i, route in enumerate(batched_routes):
        if route:
            route_points = np.array([point[0] for point in route])
            plt.plot(route_points[:, 0], route_points[:, 1], linestyle='-', marker='o', label=f'Drone {i + 1} Route')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title('Drone Delivery Route Planning')
    plt.grid()
    plt.show()

# 绘制最终的配送路径
plot_routes(delivery_center_locations, orders, batched_routes, delivery_center_names)