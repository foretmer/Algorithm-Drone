import numpy as np
import random
import math
import matplotlib.pyplot as plt
from pylab import mpl
from matplotlib.patches import Circle
import time
import pickle

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]


# 参数
num_centers = 5          # 配送中心数量
num_points = 60          # 卸货点数量
map_size = 40            # 地图尺寸 单位为公里
t = 30                   # 时间段间隔，单位为分钟
n = 10                   # 无人机最大可承载订单数量
max_distance = 20        # 无人机最大飞行距离，单位为公里
speed = 60               # 无人机飞行速度，单位为公里/小时
time_limit = 24 * 60     # 一天的时间限制，单位为分钟（24小时）
population_size = 500    # 遗传算法种群规模
generations = 30         # 遗传算法运行的迭代次数
mutation_rate = 0.3      # 变异率
crossover_rate = 0.9     # 交叉率

def save_map(centers, points, distance_matrix, filename):
    with open(filename, 'wb') as f:
        pickle.dump((centers, points, distance_matrix), f)


def load_map(filename):
    with open(filename, 'rb') as f:
        centers, points, distance_matrix = pickle.load(f)
    return centers, points, distance_matrix


def generate_map(num_centers, num_points, map_size, max_distance):
    # 在地图的四个角和中心生成配送中心
    centers = [
        (0, (max_distance/2, max_distance/2)),
        (1, (max_distance/2, map_size-max_distance/2)),
        (2, (map_size-max_distance/2, max_distance/2)),
        (3, (map_size-max_distance/2, map_size-max_distance/2)),
        (4, (map_size / 2, map_size / 2))
    ]

    points = []
    point_id = num_centers  # 编号从配送中心之后开始

    for center in centers:
        center_id, center_coord = center
        num_points_per_center = num_points // num_centers  # 平均分配给每个配送中心的卸货点数量
        for _ in range(num_points_per_center):
            while True:
                # 在配送中心为圆心，最大飞行距离的一半为半径的范围内生成卸货点
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(1, max_distance / 2)
                point_x = center_coord[0] + radius * np.cos(angle)
                point_y = center_coord[1] + radius * np.sin(angle)

                # 确保生成的点在地图范围内
                if 0 <= point_x <= map_size and 0 <= point_y <= map_size:
                    point = (point_id, (point_x, point_y))
                    points.append(point)
                    point_id += 1
                    break

    # 计算所有点对之间的距离
    all_points = centers + points
    num_all_points = len(all_points)
    distance_matrix = np.zeros((num_all_points, num_all_points))

    for i in range(num_all_points):
        for j in range(i + 1, num_all_points):
            point1 = all_points[i][1]
            point2 = all_points[j][1]
            distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return centers, points, distance_matrix


# 生成订单函数，使用模拟时间
def generate_orders(current_time, points):
    orders = []
    for point in points:
        num_orders = random.randint(0, 3)  # 每个卸货点都至少有一个订单
        for _ in range(num_orders):
            priority = random.choice(['一般', '较紧急', '紧急'])
            order_time = current_time
            if priority == '一般':
                deadline = current_time + 180  # 3小时
            elif priority == '较紧急':
                deadline = current_time + 90  # 1.5小时
            else:  # '紧急'
                deadline = current_time + 30  # 30分钟
            orders.append((point, priority, order_time, deadline))  # 添加订单信息
    return orders


def process_orders(new_orders, remaining_orders, current_time):
    current_to_deliver = []
    updated_remaining_orders = []  # 复制剩余订单列表，防止直接修改原始列表

    # 将剩余订单中的紧急和较紧急订单加入到当前配送列表,其余的加入待配送列表
    for order in remaining_orders:
        point, priority, order_time, deadline = order
        time_left = deadline - current_time
        urgency_level = calculate_urgency(time_left)
        if urgency_level in ['紧急', '较紧急']:
            current_to_deliver.append(order)
        else:
            updated_remaining_orders.append(order)

    # 根据紧急程度决定本次配送和更新剩余订单列表
    for order in new_orders:
        point, priority, order_time, deadline = order
        time_left = deadline - current_time
        urgency_level = calculate_urgency(time_left)
        if urgency_level in ['紧急', '较紧急']:
            current_to_deliver.append(order)
        else:
            updated_remaining_orders.append(order)
    if current_time == time_limit - t:
        current_to_deliver.extend(updated_remaining_orders)
    return current_to_deliver, updated_remaining_orders


def calculate_urgency(time_left):
    # 根据截止时间剩余的时间长度计算紧急程度
    if time_left <= 30:
        return '紧急'
    elif time_left <= 90:
        return '较紧急'
    else:
        return '一般'


def maintain_order_counts(centers, points, orders):
    # 初始化订单数量数组，配送中心的订单数量为0
    num_all_points = len(centers) + len(points)
    order_counts = [0] * num_all_points

    # 记录每个点的订单数量
    for order in orders:
        point_id = order[0][0]
        order_counts[point_id] += 1

    return order_counts


def print_individual(individual):
    for path in individual:
        path_str = " -> ".join(str(point[0]) for point in path)
        path_distance = sum(distance_matrix[path[i][0]][path[i+1][0]] for i in range(len(path) - 1))
        print(f"路径: {path_str}，长度: {path_distance:.2f}")


def initialize_population(centers, points, population_size, max_distance, order_counts, n):
    population = []
    for i in range(population_size):
        individual = [[] for _ in centers]  # 为每个配送中心初始化一条路径
        population.append([path for path in individual_fix(individual, order_counts) if path])
        # print(population[i])
    return population


def fitness(individual):
    total_distance = 0
    penalty = 0  # 添加 penalty 变量
    for path in individual:
        path_distance = 0
        path_orders = 0
        for i in range(len(path) - 1):
            path_distance += distance_matrix[path[i][0]][path[i + 1][0]]
            path_orders += order_counts[path[i][0]]
        if path_distance > max_distance:
            penalty += 1000000  # 如果路径超过最大飞行距离，给予较大的惩罚
        if path_distance > n:
            penalty += 1000000
        total_distance += path_distance

    return 1 / (total_distance + penalty + 0.0000000000001)  # 加入惩罚项的适应度计算


# 选择操作
def selection(population, fitnesses):
    selected_indices = np.random.choice(range(len(population)), size=2, p=fitnesses / fitnesses.sum(), replace=False)
    return population[selected_indices[0]], population[selected_indices[1]]


# 交叉操作
def pmx_crossover(parent1, parent2):
    def pmx_single(parent_a, parent_b):
        size = min(len(parent_a), len(parent_b))
        child = [None] * size
        # 选择两个交叉点
        point1, point2 = sorted(random.sample(range(1, size - 1), 2))

        # 将交叉点之间的部分复制到子代
        child[point1:point2] = parent_a[point1:point2]

        # 处理交叉点之外的部分
        for i in range(point1, point2):
            if parent_b[i] not in child:
                j = i
                while point1 <= j < point2:
                    j = parent_b.index(parent_a[j])
                child[j] = parent_b[i]

        # 处理剩余位置
        for i in range(size):
            if child[i] is None:
                child[i] = parent_b[i]

        return child if len(child) != 0 else parent_a

    if random.random() < crossover_rate:
        children1 = []
        children2 = []

        for i in range(min(len(parent1), len(parent2))):
            parent1_nodes = set([node for node in parent1[i] if node[0] != parent1[i][0][0]])
            parent2_nodes = set([node for node in parent2[i] if node[0] != parent2[i][0][0]])

            if parent1_nodes == parent2_nodes:  # 只有节点集相同时才执行交叉
                if len(parent1[i]) > 3 and len(parent2[i]) > 3:
                    child1 = pmx_single(parent1[i], parent2[i])
                    child2 = pmx_single(parent2[i], parent1[i])
                else:
                    child1 = parent1[i]
                    child2 = parent2[i]
                if child1:
                    children1.append(child1)
                if child2:
                    children2.append(child2)
            else:
                children1.append(parent1[i])
                children2.append(parent2[i])
        children1 = [child for child in children1 if len(child) > 2]
        children2 = [child for child in children2 if len(child) > 2]
        # 确保所有路径以配送中心结束
        for child in children1:
            if child[-1][0] != child[0][0]:
                child.append(child[0])
        for child in children2:
            if child[-1][0] != child[0][0]:
                child.append(child[0])

        return individual_fix(children1, order_counts), individual_fix(children2, order_counts)
    else:
        return individual_fix(parent1, order_counts), individual_fix(parent2, order_counts)


def individual_fix(individual, order_counts):
    # 移除超出最大飞行距离或最大载货量的部分
    for path in individual:
        if len(path) > 2:
            path_distance = sum(distance_matrix[path[i][0]][path[i+1][0]] for i in range(len(path) - 1))
            path_orders = sum(order_counts[point[0]] for point in path)
            if path_distance > max_distance or path_orders > n:  # 超出限制
                excess_distance = path_distance - max_distance
                excess_orders = path_orders - n
                while (excess_distance > 0 or excess_orders > 0) and len(path) > 2:  # 循环删除节点直至路径合法
                    second_last_point = path[-2]
                    path.remove(second_last_point)
                    excess_distance -= distance_matrix[second_last_point[0]][path[-1][0]]
                    excess_orders -= order_counts[second_last_point[0]]

    # 检查是否有卸货点未配送
    unvisited_points = [point for point in points if point not in sum(individual, [])]
    random.shuffle(unvisited_points)
    new_path = [[center] for center in centers]  # 为每个配送中心初始化一条路径
    while unvisited_points:
        point = unvisited_points.pop(0)
        if order_counts[point[0]] == 0:
            continue
        # 找到最近的配送中心
        nearest_center = min(centers, key=lambda center: distance_matrix[center[0]][point[0]])
        for path in new_path:
            if path[0][0] != nearest_center[0]:
                continue
            current_distance = sum(distance_matrix[path[i][0]][path[i+1][0]] for i in range(len(path) - 1))
            distance_to_point = distance_matrix[path[-1][0]][point[0]]
            current_orders = sum(order_counts[point[0]] for point in path)
        # 判断加入当前点后是否超过最大距离
            if current_distance + distance_to_point + distance_matrix[point[0]][nearest_center[0]] <= max_distance and current_orders + order_counts[point[0]] <= n:
                path.append(point)
                break
            else:
                new_path.append([nearest_center, point])  # 开始新路径
                break
        # new_path[path_index] = path

    for path in new_path:
        if len(path) > 1:
            individual.append(path)
    # 确保所有路径以配送中心结束
    for path in individual:
        if path and path[-1][0] != path[0][0]:
            path.append(path[0])
    # 使用集合去重
    individual_set = {tuple(path) for path in individual}
    individual = [list(path) for path in individual_set]
    individual = [path for path in individual if len(path) > 2]
    return individual


def mutate(individual):
    for i, path in enumerate(individual):
        if random.random() < mutation_rate and len(path) > 2:
            mutation_type = random.randint(1, 4)  # 随机选择变异类型
            unloading_points = [idx for idx, point in enumerate(path) if idx != 0 and idx != len(path) - 1]
            if mutation_type == 1 and len(unloading_points) > 1:  # 路径交换，确保至少有两个卸货点
                swap_points = random.sample(unloading_points, 2)
                path[swap_points[0]], path[swap_points[1]] = path[swap_points[1]], path[swap_points[0]]
            elif mutation_type == 2 and len(unloading_points) > 1:  # 路径倒置，确保至少有两个卸货点
                reverse_start = random.choice(unloading_points)
                valid_end_points = [idx for idx in unloading_points if idx > reverse_start]
                if valid_end_points:  # 确保有合法的结束点
                    reverse_end = random.choice(valid_end_points)
                    path[reverse_start:reverse_end + 1] = reversed(path[reverse_start:reverse_end + 1])
            elif mutation_type == 3:
                individual[i] = []
            elif mutation_type == 4 and len(unloading_points) > 0:  # 随机选择一个点插入到另一条路径
                selected_point_index = random.choice(unloading_points)
                selected_point = path.pop(selected_point_index)

                # 找到该点所属的配送中心
                center_id = path[0][0]
                # 找到所有其他由该配送中心负责的路径
                other_paths = [p for p in individual if len(p) > 2 and p[0][0] == center_id and p != path]

                if other_paths:
                    # 随机选择一个目标路径
                    target_path = random.choice(other_paths)
                    # 随机选择一个插入位置
                    insert_position = random.randint(1, len(target_path) - 1)
                    target_path.insert(insert_position, selected_point)
                else:
                    # 如果没有其他路径，则将点插回原路径的随机位置
                    insert_position = random.randint(1, len(path) - 1)
                    path.insert(insert_position, selected_point)
    # 使用 individual_fix 函数修复路径
    individual = individual_fix(individual, order_counts)
    return individual


def genetic_algorithm(centers, points, population_size, generations, max_distance, order_counts, n):
    population = initialize_population(centers, points, population_size, max_distance, order_counts, n)  # 初始化种群
    best_individual = None  # 初始化最佳个体为空
    best_fitness = float('-inf')  # 初始化最佳适应度为负值

    for generation in range(generations):
        fitnesses = np.array([fitness(ind) for ind in population])  # 计算适应度函数

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitnesses)  # 轮盘赌选择两个父本个体
            child1, child2 = pmx_crossover(parent1, parent2)  # 对父本个体应用部分匹配交叉得到两个孩子个体
            mutate(child1)  # 对孩子个体进行变异
            mutate(child2)
            new_population.extend([child1, child2])  # 繁育出新种群

        population = new_population

        current_best = population[np.argmax(fitnesses)]
        current_fitness = max(fitnesses)

        if current_fitness > best_fitness:  # 更新最佳个体
            best_fitness = current_fitness
            best_individual = current_best
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    return [path for path in best_individual if len(path) > 2]  # 返回最优个体


# 绘制地图
def plot_map(centers, points, paths, max_distance):
    plt.figure(figsize=(12, 12))

    # 绘制配送中心
    centers_x, centers_y = zip(*[center[1] for center in centers])
    centers_labels = [center[0] for center in centers]
    plt.scatter(centers_x, centers_y, c='blue', marker='s', s=100, label='配送中心')
    for i, txt in enumerate(centers_labels):
        plt.annotate(txt, (centers_x[i], centers_y[i]), fontsize=12, fontweight='bold', ha='right')

    # 绘制配送中心的服务范围圆
    for center in centers:
        circle = Circle(center[1], max_distance / 2, color='blue', alpha=0.1, linestyle='--')
        plt.gca().add_patch(circle)

    # 绘制卸货点
    points_x, points_y = zip(*[point[1] for point in points])
    points_labels = [point[0] for point in points]
    plt.scatter(points_x, points_y, c='green', marker='o', s=60, label='卸货点')
    for i, txt in enumerate(points_labels):
        plt.annotate(txt, (points_x[i], points_y[i]), fontsize=10, ha='right')

    # 绘制路径
    colors = plt.cm.rainbow(np.linspace(0, 1, len(paths)))  # 使用不同颜色绘制每条路径
    for path, color in zip(paths, colors):
        path_x, path_y = zip(*[point[1] for point in path])
        plt.plot(path_x, path_y, '-', color=color, linewidth=2, alpha=0.7)
        for point in path:
            plt.annotate(point[0], point[1], fontsize=8, ha='center')

    plt.title('无人机配送路径规划', fontsize=16, fontweight='bold')
    plt.xlabel('X 坐标', fontsize=14)
    plt.ylabel('Y 坐标', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis((0, map_size, 0, map_size))
    plt.show()


def print_best_individual(best_result, orders):
    print("最优路径和路径长度：")
    for path in best_result:
        path_str = " -> ".join(str(point[0]) for point in path)
        path_distance = sum(distance_matrix[path[i][0]][path[i + 1][0]] for i in range(len(path) - 1))
        print(f"路径: {path_str}，长度: {path_distance:.2f}")

        # 输出订单信息
        for point in path:
            point_id = point[0]
            if point_id < len(centers):
                continue  # 跳过配送中心

            point_orders = [order for order in orders if order[0][0] == point_id]
            for order in point_orders:
                order_priority = order[1]
                order_time = order[2]
                order_deadline = order[3]
                print(f"  点 {point_id} 的订单 - 优先级: {order_priority}, 下单时间: {order_time // 60} 小时 {order_time % 60} 分钟, 截止时间: {order_deadline // 60} 小时 {order_deadline % 60}")


# 生成地图
# centers, points, distance_matrix = generate_map(num_centers, num_points, map_size, max_distance)  # 随机生成地图
# save_map(centers, points, distance_matrix, 'map_data.pkl')  # 保存地图文件
centers, points, distance_matrix = load_map('map_data.pkl')
orders = []
total_daily_distance = 0
todo_list = []
for current_time in range(0, time_limit, t):
    # 生成当前时间段的订单
    new_orders = generate_orders(current_time, points)
    start_time = time.time()  # 获取开始时间
    current_orders, todo_list = process_orders(new_orders, todo_list, current_time)
    # 计算订单数量统计
    order_counts = maintain_order_counts(centers, points, current_orders)
    print(f"当前时间： {current_time} order_counts: {order_counts}")
    # 运行遗传算法优化路径规划
    best_result = genetic_algorithm(centers, points, population_size, generations, max_distance, order_counts, n)
    # 获取结束时间
    end_time = time.time()
    use_time = end_time - start_time
    print(f"决策用时: {use_time:.2f}s")
    current_best_distance = sum(distance_matrix[path[i][0]][path[i + 1][0]] for path in best_result for i in range(len(path) - 1))
    total_daily_distance += current_best_distance
    # 输出最优路径
    hours = current_time // 60
    minutes = current_time % 60
    print(f"当前时间: {hours} 小时 {minutes} 分钟")
    print_best_individual(best_result, current_orders)
    # 绘制最佳路径
    plot_map(centers, points, best_result, max_distance)

print("所有时间段订单处理完毕。")
print(f"总路径长度: {total_daily_distance:.2f}")