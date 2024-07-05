import matplotlib.pyplot as plt
import numpy as np
import random
import math
from collections import defaultdict
from pylab import mpl

# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
num_centers = 5
max_distance_per_trip = 20  # 最大飞行距离为20公里
drone_speed_kmph = 60  # 无人机速度60公里/小时
max_orders_per_trip = 5  # 无人机一次最多携带5个物品
radius = 10  # 每个配送中心的服务半径

# 配送中心坐标
np.random.seed(32)
center_coords = np.random.rand(num_centers, 2) * 40


# 生成卸货点
def generate_drop_off_points(num_drop_off_points, center_coords, radius):
    drop_off_coords = []
    drop_off_index_map = {}
    for i in range(num_drop_off_points):
        center_idx = np.random.randint(0, num_centers)
        center_x, center_y = center_coords[center_idx]
        angle = np.random.rand() * 2 * np.pi
        distance = np.random.rand() * radius
        drop_off_x = center_x + distance * np.cos(angle)
        drop_off_y = center_y + distance * np.sin(angle)
        drop_off_coords.append((drop_off_x, drop_off_y))
        drop_off_index_map[i + 1] = (drop_off_x, drop_off_y)  # 使用1-based索引
    return np.array(drop_off_coords), drop_off_index_map


drop_off_coords, drop_off_index_map = generate_drop_off_points(40, center_coords, radius)


# 生成订单
def generate_orders(drop_off_coords, t):
    orders = []
    for i, location in enumerate(drop_off_coords):
        num_orders = random.randint(0, 5)
        for _ in range(num_orders):
            priority = random.choice(['一般', '较紧急', '紧急'])
            if priority == '一般':
                delivery_time =  3  # 随机生成订单的配送时间
            elif priority == '较紧急':
                delivery_time =  1.5  # 随机生成订单的配送时间
            elif priority == '紧急':
                delivery_time =  0.5  # 随机生成订单的配送时间
            orders.append({'location': location, 'priority': priority, 'delivery_time': delivery_time, 'drop_off_id': i + 1})
    # print("生成的订单：")
    # for n in orders:
    #     print(n)
    return orders


# 计算两点之间的距离
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 贪心算法
def greedy_path_planning(center_coords, drop_off_coords, orders, t, max_load=max_orders_per_trip,
                         max_distance=max_distance_per_trip):
    paths = defaultdict(list)
    total_distance = 0
    deferred_orders = []

    # 按配送时间排序订单
    orders.sort(key=lambda x: ( x['delivery_time']))

    # 将订单分配到最近的配送中心
    center_orders = defaultdict(list)
    for order in orders:
        min_distance = float('inf')
        nearest_center = None
        for center_idx, center in enumerate(center_coords):
            distance = calculate_distance(center, order['location'])
            if distance < min_distance:
                min_distance = distance
                nearest_center = center_idx
        center_orders[nearest_center].append(order)


    # # 打印每个配送中心被分配的订单
    # for center_idx, orders in center_orders.items():
    #     print(f'配送中心{center_idx + 1}分配的订单:')
    #     print(len(orders))
    #     for order in orders:
    #         drop_off_id = np.argmin(np.sum((drop_off_coords - order['location']) ** 2, axis=1)) + 1
    #         print(f"  订单ID: {drop_off_id}, 优先级: {order['priority']}, 生成时间: {order['generated_time']}")



    # 为每个配送中心规划路径
    for center_idx, center_orders in center_orders.items():
        remaining_orders = center_orders
        while remaining_orders:
            current_position = center_coords[center_idx]
            current_load = 0
            current_distance = 0
            path = [current_position]
            pending_orders = []
            non_urgent_orders = []

            for order in remaining_orders:

                order_distance = calculate_distance(current_position, order['location'])
                return_distance = calculate_distance(order['location'], current_position)

                if current_load < max_load and (current_distance + order_distance + return_distance <= max_distance):
                    if order['delivery_time'] >= 1.5 :
                        non_urgent_orders.append(order)

                    path.append(order['location'])
                    current_position = order['location']
                    current_load += 1
                    current_distance += order_distance
                else:
                    pending_orders.append(order)

            intervals = 23
            # print("不紧急数目",len(non_urgent_orders))
            # print("当前数目",current_load)
            # print("距离",current_distance)


            # 如果路径距离小于10公里且全都是不紧急订单，则推迟配送
            if current_distance < 10 and len(non_urgent_orders) == current_load and t < intervals * 30:
                deferred_orders.extend(non_urgent_orders)

            else:
                path.append(center_coords[center_idx])
                if len(path) > 2:  # 确保路径中包含至少一个卸货点
                    distance = current_distance + calculate_distance(current_position, center_coords[center_idx])
                    paths[center_idx].append((path, distance))
                    total_distance += current_distance + calculate_distance(current_position, center_coords[center_idx])


            remaining_orders = pending_orders

    return paths, total_distance, deferred_orders


# 模拟每30分钟生成订单并进行路径规划
def simulate_day(center_coords, drop_off_coords, intervals=24):
    total_paths = defaultdict(list)
    total_distance = 0
    deferred_orders = []
    time_period_paths = {}

    for t in range(0, intervals * 30, 30):
        # 更新推迟订单的配送时间
        for order in deferred_orders:
            order['delivery_time'] -= 0.5

        # orders = generate_orders(drop_off_coords, t)
        # orders.extend(deferred_orders)
        # paths, distance, deferred_orders = greedy_path_planning(center_coords, drop_off_coords, orders, t)
        # time_period_paths[t] = paths
        #
        # # 统计每个时间段的订单和无人机信息
        # num_orders = len(orders)
        # num_processed_orders = num_orders - len(deferred_orders)
        # num_drones = sum(len(paths[center]) for center in paths)

        # for center, center_paths in paths.items():
        #     total_distance += sum(distance for _, distance in center_paths)


        orders = generate_orders(drop_off_coords, t)
        num = len(orders)

        # print(f'时间{t}分钟生成的订单: {orders}')
        # print(len(orders))
        orders.extend(deferred_orders)
        paths, distance, deferred_orders = greedy_path_planning(center_coords, drop_off_coords, orders, t)
        # print(len(orders))
        # print(len(deferred_orders))
        time_period_paths[t] = paths
        for center, center_paths in paths.items():
            total_paths[center].extend(center_paths)

        # print(orders)
        # print(deferred_orders)
        # 统计每个时间段的订单和无人机信息
        num_orders = len(orders)
        num_processed_orders = num_orders - len(deferred_orders)
        # print(paths)
        num_drones = sum(len(paths[center]) for center in paths)
        print(
            f'时间{t}分钟统计：生成了{num}个订单，累计有{num_orders}个订单，处理了{num_processed_orders}个订单，一共派出了{num_drones}架无人机。')
        total_distance=0
        for center, center_paths in paths.items():
            total_distance += sum(distance for _, distance in center_paths)
            for i, (path, distance) in enumerate(center_paths):
                path_str = ' -> '.join(
                    [f'配送中心{center + 1}'] + [f'卸货点{np.argmin(np.sum((drop_off_coords - p) ** 2, axis=1)) + 1}'
                                                 for p in path[1:-1]] + [f'配送中心{center + 1}'])
                print(f'  无人机{i + 1}的路径: {path_str}, 路径距离: {distance:.2f}公里')

        # 打印t时刻的总路径长度
        print(f'时间{t}分钟总路径长度: {total_distance:.2f}公里')

    return total_paths, total_distance, time_period_paths


# 执行模拟
total_paths, total_distance, time_period_paths = simulate_day(center_coords, drop_off_coords)

# 可视化结果（此部分略去，请根据需要进行可视化）
# 可视化结果
def visualize_paths(center_coords, drop_off_coords, drop_off_index_map, paths, time_period):
    plt.figure(figsize=(10, 8))
    plt.scatter(center_coords[:, 0], center_coords[:, 1], color='blue', marker='s', s=100, label='配送中心')

    # 所有卸货点
    plt.scatter(drop_off_coords[:, 0], drop_off_coords[:, 1], color='grey', marker='o', s=50, label='卸货点')

    # 标注配送中心
    for i, (x, y) in enumerate(center_coords):
        plt.text(x, y, f'中心{i + 1}', fontsize=12, ha='right')
        circle = plt.Circle((x, y), radius, color='blue', fill=False, linestyle='--')
        plt.gca().add_patch(circle)

    # 绘制无人机路径
    colors = ['green', 'purple', 'orange', 'cyan','red']
    for center_idx, center_paths in paths.items():
        for path, _ in center_paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], color=colors[center_idx % len(colors)], linestyle='-', marker='o')
            for i, (x, y) in enumerate(path):
                if i == 0:
                    plt.text(x, y, f'中心{center_idx + 1}', fontsize=12, ha='right')
                else:
                    drop_off_id = np.argmin(np.sum((drop_off_coords - (x, y)) ** 2, axis=1)) + 1
                    plt.text(x, y, f'{drop_off_id}', fontsize=8, ha='right')

    plt.title(f'配送中心和无人机路径分布图 (时间段: {time_period} 分钟)')
    plt.xlabel('X 坐标 (km)')
    plt.ylabel('Y 坐标 (km)')
    plt.grid(True)
    plt.xlim(-10, 50)
    plt.ylim(0, 50)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()


# 绘制每个时间段的路径规划
for time_period, paths in time_period_paths.items():
    visualize_paths(center_coords, drop_off_coords, drop_off_index_map, paths, time_period)

