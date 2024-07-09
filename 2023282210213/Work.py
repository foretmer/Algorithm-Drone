import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from math import sqrt
import random

# 假设条件
n = 5  # 无人机一次最多携带5个物品
max_distance = 20  # 无人机一次飞行最远距离为20公里
drone_speed = 60  # 无人机速度为60公里/小时
time_interval = 15  # 时间间隔t分钟
total_time = 24 * 60  # 一天的总时间

# 配送中心和卸货点的位置
delivery_centers = [(2, 3), (9, 6), (7, 10), (5, 5)]
unloading_points = [(0, 0), (5, 1), (8, 8), (6, 6), (1, 7), (3, 5), (7, 2), (4, 4), (9, 3), (5, 8), (2, 9)]


# 订单类
class Order:
    def __init__(self, priority, location, timestamp):
        self.priority = priority
        self.location = location
        self.timestamp = timestamp

    def to_dict(self):
        return {
            "priority": self.priority,
            "location_x": self.location[0],
            "location_y": self.location[1],
            "timestamp": self.timestamp
        }


# 生成订单
def generate_orders(timestamp):
    num_orders = np.random.randint(0, 6)  # 每次生成0-5个订单
    orders = []
    for _ in range(num_orders):
        priority = np.random.choice(['一般', '较紧急', '紧急'], p=[0.5, 0.3, 0.2])
        location = unloading_points[np.random.randint(len(unloading_points))]
        orders.append(Order(priority, location, timestamp))
    return orders


# 计算距离
def calculate_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 规划路径（贪心算法）
def plan_path(orders):
    orders = sorted(orders, key=lambda x: x.priority)
    paths = defaultdict(list)

    for center in delivery_centers:
        remaining_orders = orders[:]
        while remaining_orders:
            current_location = center
            current_path = []
            current_distance = 0

            while remaining_orders and current_distance < max_distance:
                next_order = min(remaining_orders, key=lambda x: calculate_distance(current_location, x.location))
                distance_to_order = calculate_distance(current_location, next_order.location)

                if current_distance + distance_to_order + calculate_distance(next_order.location,
                                                                             center) <= max_distance:
                    current_path.append(next_order)
                    current_distance += distance_to_order
                    current_location = next_order.location
                    remaining_orders.remove(next_order)
                else:
                    break

            paths[center].append(current_path)

    return paths


# 路径优化（模拟退火算法）
def simulated_annealing(path, temperature=10000, cooling_rate=0.003):
    def calculate_total_distance(path):
        total_distance = 0
        current_location = delivery_centers[0]
        for order in path:
            total_distance += calculate_distance(current_location, order.location)
            current_location = order.location
        total_distance += calculate_distance(current_location, delivery_centers[0])
        return total_distance

    def swap_two_orders(path):
        if len(path) < 2:
            return path
        new_path = path[:]
        idx1, idx2 = random.sample(range(len(new_path)), 2)
        new_path[idx1], idx2 = new_path[idx2], new_path[idx1]
        return new_path

    current_path = path
    best_path = path
    best_distance = calculate_total_distance(best_path)

    while temperature > 1:
        new_path = swap_two_orders(current_path)
        current_distance = calculate_total_distance(current_path)
        new_distance = calculate_total_distance(new_path)

        if new_distance < current_distance or random.random() < np.exp((current_distance - new_distance) / temperature):
            current_path = new_path

        if new_distance < best_distance:
            best_path = new_path
            best_distance = new_distance

        temperature *= 1 - cooling_rate

    return best_path


# 生成并规划路径和优化路径
def generate_and_plan_paths():
    print("生成订单数据...")
    all_orders = []
    for timestamp in range(0, total_time, time_interval):
        new_orders = generate_orders(timestamp)
        all_orders.extend(new_orders)

    print("保存订单数据到CSV文件...")
    order_data = pd.DataFrame([order.to_dict() for order in all_orders])
    order_data.to_csv('orders.csv', index=False, encoding='utf-8-sig')  # 使用UTF-8带BOM

    print("规划路径...")
    paths = plan_path(all_orders)

    print("优化路径...")
    for center, path_list in paths.items():
        for i, path in enumerate(path_list):
            optimized_path = simulated_annealing(path)
            paths[center][i] = optimized_path

    return paths


# 可视化配送中心和卸货点的位置
def visualize_centers_and_points(delivery_centers, unloading_points):
    plt.figure(figsize=(10, 10))

    # 绘制配送中心
    for center in delivery_centers:
        plt.scatter(center[0], center[1], c='red', marker='o', s=100, label='配送中心')

    # 绘制卸货点
    for point in unloading_points:
        plt.scatter(point[0], point[1], c='blue', marker='x', s=100, label='卸货点')

    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('配送中心与卸货点位置')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


# 可视化路径
def visualize_paths(paths):
    plt.figure(figsize=(10, 10))

    # 绘制配送中心和卸货点
    for center in delivery_centers:
        plt.scatter(center[0], center[1], c='red', marker='o', s=100, label='配送中心')

    for point in unloading_points:
        plt.scatter(point[0], point[1], c='blue', marker='x', s=100, label='卸货点')

    # 绘制路径
    colors = ['green', 'purple', 'orange', 'brown', 'pink']
    for center, path_list in paths.items():
        for i, path in enumerate(path_list):
            path_points = [center] + [order.location for order in path] + [center]
            path_points = np.array(path_points)
            plt.plot(path_points[:, 0], path_points[:, 1], c=colors[i % len(colors)], label=f'路径{i + 1}')

    plt.legend()
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('无人机配送路径规划')
    plt.grid()
    plt.show()


# 主函数
if __name__ == "__main__":
    print("可视化配送中心与卸货点的位置...")
    visualize_centers_and_points(delivery_centers, unloading_points)

    print("生成并规划路径和优化路径...")
    paths = generate_and_plan_paths()

    print("可视化路径...")
    visualize_paths(paths)
    print("完成。")
