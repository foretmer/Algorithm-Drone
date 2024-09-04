import numpy as np
import random
import matplotlib.pyplot as plt

# 初始化参数
alpha = 1.0  # 信息素重要程度
beta = 2.0  # 启发式信息重要程度
rho = 0.7  # 信息素挥发系数
num_ants = 10  # 蚂蚁数量
num_iterations = 50  # 迭代次数
t = 10  # 时间离散化步长（分钟）
n = 5  # 无人机一次最多携带的物品数量
max_distance = 20  # 无人机一次飞行最远路程（公里）
speed = 60  # 无人机速度（公里/小时）

# 假设配送中心和卸货点的位置
centers = [(0, 0), (10, 10), (12, 6), (3, 1), (5, 5)]  # 5个配送中心
delivery_points = [(2, 3),(4, 1), (5, 2), (6, 7),(7, 1), (8, 4),(9, 9), (10, 15),(11, 12), (12, 5), (13, 12),
                   (1, 10),(2, 2), (3, 6), (4, 7),(5, 5), (6, 15),(2, 9), (10, 11),(12, 1), (7, 10), (3, 12),]  # 卸货点位置

# 计算距离矩阵
def compute_distance_matrix(centers, delivery_points):
    points = centers + delivery_points
    num_points = len(points)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            else:
                distance_matrix[i][j] = np.inf  # 避免自身距离为零的情况
    return distance_matrix

# 初始化信息素矩阵
def initialize_pheromone_matrix(num_nodes):
    return np.ones((num_nodes, num_nodes))

# 选择下一个配送点
def select_next_point(current_point, visited, pheromone_matrix, distance_matrix, alpha, beta):
    """
    根据当前节点、已访问节点、信息素矩阵和距离矩阵，选择下一个配送点。

    参数:
    current_point (int): 当前所在的节点索引。
    visited (set): 已访问节点的集合。
    pheromone_matrix (numpy.ndarray): 信息素矩阵。
    distance_matrix (numpy.ndarray): 距离矩阵。
    alpha (float): 信息素的重要程度。
    beta (float): 启发式信息的重要程度。

    返回:
    int: 下一个配送点的索引。
    """
    num_points = len(distance_matrix)
    probabilities = np.zeros(num_points)
    total_prob = 0.0
    for i in range(num_points):
        if i not in visited:
            if distance_matrix[current_point][i] > 0:
                probabilities[i] = (pheromone_matrix[current_point][i] ** alpha) * ((1.0 / distance_matrix[current_point][i]) ** beta)
            else:
                probabilities[i] = 0
            total_prob += probabilities[i]

    if total_prob == 0:
        return None

    probabilities /= total_prob
    next_point = np.random.choice(range(num_points), p=probabilities)
    return next_point if next_point not in visited else None

# 更新信息素矩阵
def update_pheromone_matrix(pheromone_matrix, all_routes, distance_matrix, rho):
    """
    根据所有蚂蚁的路径更新信息素矩阵。

    参数:
    pheromone_matrix (numpy.ndarray): 原始信息素矩阵。
    all_routes (list): 所有蚂蚁的路径列表。
    distance_matrix (numpy.ndarray): 距离矩阵。
    rho (float): 信息素挥发系数。

    返回:
    numpy.ndarray: 更新后的信息素矩阵。
    """
    num_nodes = len(pheromone_matrix)
    delta_pheromone = np.zeros((num_nodes, num_nodes))
    for route in all_routes:
        route_length = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))
        if route_length > 0:
            for i in range(len(route) - 1):
                delta_pheromone[route[i]][route[i+1]] += 1.0 / route_length
    pheromone_matrix = (1 - rho) * pheromone_matrix + delta_pheromone
    return pheromone_matrix

# 选择最佳路径
def select_best_route(all_routes, distance_matrix):
    """
    从所有路径中选择最短的路径。

    参数:
    all_routes (list): 所有蚂蚁的路径列表。
    distance_matrix (numpy.ndarray): 距离矩阵。

    返回:
    list: 最短路径。
    """
    best_route = None
    best_length = float('inf')
    for route in all_routes:
        route_length = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))
        if route_length < best_length:
            best_length = route_length
            best_route = route
    return best_route

# 生成订单
def generate_orders(delivery_points, max_orders=5):
    """
    模拟生成订单，每个配送点随机生成0到5个订单，并分配优先级。

    参数:
    delivery_points (list): 配送点列表。
    priorities (list): 订单优先级列表。

    返回:
    list: 生成的订单列表，格式为 (配送点, 优先级)。
    """
    orders = []
    for i, point in enumerate(delivery_points):
        num_orders = random.randint(0, max_orders)
        for _ in range(num_orders):
            priority = random.choice(['一般', '较紧急', '紧急'])
            orders.append((i + len(centers), priority))  # 订单格式：(卸货点索引, 优先级)
    return orders

# 主路径规划算法
def ant_colony_optimization(centers, delivery_points, num_ants, num_iterations, alpha, beta, rho):
    distance_matrix = compute_distance_matrix(centers, delivery_points)
    num_nodes = len(centers) + len(delivery_points)
    pheromone_matrix = initialize_pheromone_matrix(num_nodes)
    
    best_route = None
    best_length = float('inf')

    for iteration in range(num_iterations):
        all_routes = []
        
        for ant in range(num_ants):
            visited = set()
            current_point = random.choice(range(len(centers)))  # Start from a random center
            route = [current_point]
            visited.add(current_point)
            
            while len(visited) < num_nodes:
                next_point = select_next_point(current_point, visited, pheromone_matrix, distance_matrix, alpha, beta)
                if next_point is None:
                    break
                route.append(next_point)
                visited.add(next_point)
                current_point = next_point
            
            route.append(route[0])  # Return to the starting center
            all_routes.append(route)
        
        pheromone_matrix = update_pheromone_matrix(pheromone_matrix, all_routes, distance_matrix, rho)
        current_best_route = select_best_route(all_routes, distance_matrix)
        current_best_length = sum(distance_matrix[current_best_route[i]][current_best_route[i+1]] for i in range(len(current_best_route) - 1))
        
        if current_best_length < best_length:
            best_length = current_best_length
            best_route = current_best_route

    return best_route, best_length

# 模拟订单生成和路径规划
def simulate_delivery(centers, delivery_points, num_ants, num_iterations, alpha, beta, rho, t, simulation_time):
    total_time = 0
    all_orders = []

    while total_time < simulation_time:
        orders = generate_orders(delivery_points)
        all_orders.extend(orders)
        
        # 按优先级排序订单
        all_orders.sort(key=lambda x: ('一般', '较紧急', '紧急').index(x[1]))
        
        # 规划路径并分配无人机
        best_route, best_length = ant_colony_optimization(centers, delivery_points, num_ants, num_iterations, alpha, beta, rho)
        print(f"Ants num {num_ants}  Iterations num {num_iterations}")
        print(f"Best route at time {total_time} minutes: {best_route} with length {best_length}")
        filename = f"{total_time}_route"
        plot_route(best_route, centers, delivery_points, total_time, filename)
        
        # 模拟配送
        all_orders = []
        
        total_time += t

# 可视化最佳路径
def plot_route(route, centers, delivery_points, total_time,filename):
    points = centers + delivery_points
    route_points = [points[i] for i in route]

    plt.figure()
    plt.plot([p[0] for p in route_points], [p[1] for p in route_points], marker='o')
    
    for i, point in enumerate(points):
        plt.text(point[0], point[1], f"{i}", fontsize=12, ha='right')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f"Best Delivery Route Visulization at time {total_time}")
    plt.grid(True)
    plt.savefig(filename)
    # plt.show()
    plt.close() 
# 参数设置
simulation_time = 1440  # 模拟一天（1440分钟）
simulate_delivery(centers, delivery_points, num_ants, num_iterations, alpha, beta, rho, t, simulation_time)
