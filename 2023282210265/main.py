import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

n = 5  # 配货点数量
k = 10  # 每个配货点的卸货点数量
range_size = 40  # 区域范围
dock_range = 10  # 卸货点相对于配货点的最大距离

num_centers = n  # 配送中心数量
num_docks = n * k #卸货点总数量
max_orders_per_interval = 5  # 每个时间间隔内最大订单数量
time_interval = 10  # 订单生成的时间间隔 (分钟)
priority_levels = {'一般': 3, '较紧急': 1.5, '紧急': 0.5}  # 订单优先级别（小时）
max_distance = 20  # 无人机最大飞行距离 (公里)
drone_speed = 60  # 无人机速度 (公里/小时)
max_cargo = 5  # 无人机最大携带物品数量

# 计算两点之间的距离
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 在指定范围内生成配货点，确保两两之间的距离大于w
def generate_distribution_centers():
    distribution_centers = [(10, 10), (20, 20), (30, 30), (10, 30), (30, 10)]
    return distribution_centers


# 围绕每个配货点生成卸货点
def generate_delivery_docks(distribution_centers, k, dock_range):
    delivery_docks = []
    for center in distribution_centers:
        for _ in range(k):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(1, dock_range)
            new_dock = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
            # 确保卸货点在有效范围内
            if 0 <= new_dock[0] <= range_size and 0 <= new_dock[1] <= range_size:
                delivery_docks.append(new_dock)
    return delivery_docks


def read_locations_from_file():
    centers = []
    docks = []
    with open("./locations.txt", "r") as f:
        lines = f.readlines()
        read_docks = False
        for line in lines:
            line = line.strip()
            if line == "Distribution Centers:":
                read_docks = False
                continue
            elif line == "Delivery Docks:":
                read_docks = True
                continue
            if line:
                location = tuple(map(float, line.strip('()').split(', ')))
                if read_docks:
                    docks.append(location)
                else:
                    centers.append(location)
    return centers, docks


# 绘制生成的配货点和卸货点
def plot_locations(distribution_centers, delivery_docks):
    colors = ['r', 'g', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(10, 10))
    ax = plt.gca()  # 获取当前坐标轴

    # 绘制配货点并标记名称
    for idx, center in enumerate(distribution_centers):
        plt.scatter(center[0], center[1], color=colors[idx % len(colors)],
                    label=f'Distribution Center' if idx == 0 else "")
        plt.text(center[0], center[1], f'C{idx}', fontsize=12, ha='right')

        # 绘制中心的服务范围圈
        circle = patches.Circle(center, 10, color=colors[idx % len(colors)], alpha=0.2)
        ax.add_patch(circle)

    # 绘制卸货点并标记名称
    for idx, dock in enumerate(delivery_docks):
        plt.scatter(dock[0], dock[1], color='b', marker='x', label=f'Delivery Dock' if idx == 0 else "")
        plt.text(dock[0], dock[1], f'D{idx}', fontsize=8, ha='right')

    plt.legend()
    plt.title("Delivery Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, range_size)
    plt.ylim(0, range_size)
    plt.savefig("./distribution_and_delivery_locations.png")
    plt.show()
    plt.close()

# 订单生成
def generate_orders():
    orders = []
    for dock_id in range(num_docks):
        num_orders = random.randint(0, max_orders_per_interval)
        for _ in range(num_orders):
            priority = random.choice(list(priority_levels.keys()))
            orders.append((dock_id, priority))
    return orders

# 保存订单到文件
def save_orders_to_file(orders, filename):
    with open(filename, "w") as f:
        for order in orders:
            f.write(f"{order[0]},{order[1]}\n")

# 读取文件中的订单
def read_orders_from_file(filename):
    orders = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            dock_id, priority = line.strip().split(",")
            orders.append((int(dock_id), priority))
    return orders

def print_orders(orders):
    if not orders:
        print("没有生成订单。")
    else:
        print("生成的订单如下：")
        for order in orders:
            dock_id, priority = order
            print(f"卸货点ID: {dock_id}, 优先级: {priority}")

# 路径规划算法（贪心算法）
def plan_path(center, orders):
    orders = sorted(orders, key=lambda x: priority_levels[x[1]])  # 按优先级排序
    total_distance = 0
    current_location = center
    path = []
    while orders:
        trip_orders = []
        trip_distance = 0
        current_cargo = 0
        while orders and current_cargo < max_cargo:
            next_order = orders[0]
            next_location = delivery_docks[next_order[0]]
            dist_to_next = distance(current_location, next_location)
            dist_back_to_center = distance(next_location, center)
            round_trip_dist = trip_distance + dist_to_next + dist_back_to_center
            if round_trip_dist <= max_distance:
                trip_orders.append(next_order)
                trip_distance += dist_to_next
                current_location = next_location
                current_cargo += 1
                orders.pop(0)
            else:
                break
        trip_distance += distance(current_location, center)  # 回到配送中心
        total_distance += trip_distance
        path.append((trip_orders, trip_distance))
        current_location = center
    return path, total_distance

# # 初始决策系统
# def decision_system(orders):
#     total_distance = 0
#     center_orders = defaultdict(list)  # 每个配送中心分配的订单
#     for order in orders:
#         # 找到距离该订单最近的配送中心
#         closest_center = min(distribution_centers, key=lambda center: distance(center, delivery_docks[order[0]]))
#         center_orders[closest_center].append(order)

#     all_paths = defaultdict(list)  # 用于存储每个配送中心的路径
#     for center, orders in center_orders.items():
#         path, dist = plan_path(center, orders)
#         total_distance += dist
#         all_paths[center].extend(path)
#     return total_distance, center_orders, all_paths

# # 模拟一天的配送
# def simulate_day(num_intervals):
#     total_distance = 0
#     all_paths_per_interval = []  # 用于存储每个时间间隔的所有路径
#     for i in range(num_intervals):
#         # orders = generate_orders()  # 每个时间间隔生成新的订单
#         orders_file_n = "./orders/orders_"+ str(i) + ".txt"
#         orders = read_orders_from_file(orders_file_n)
#         dist, center_orders, all_paths = decision_system(orders)
#         total_distance += dist
#         all_paths_per_interval.append((center_orders, all_paths))
#     return total_distance, all_paths_per_interval

# # 时间累积决策系统
# def decision_system(all_orders, current_time):
#     total_distance = 0
#     center_orders = defaultdict(list)  # 每个配送中心分配的订单
#     pending_orders = []  # 累积的未配送订单
#     for order in all_orders:
#         dock_id, priority = order
#         time_left = priority_levels[priority] * 60 - current_time
#         if time_left > time_interval:  # 优先级较低，可以累积
#             pending_orders.append(order)
#         else:
#             closest_center = min(distribution_centers, key=lambda center: distance(center, delivery_docks[dock_id]))
#             center_orders[closest_center].append(order)

#     all_paths = defaultdict(list)  # 用于存储每个配送中心的路径
#     for center, orders in center_orders.items():
#         path, dist = plan_path(center, orders)
#         total_distance += dist
#         all_paths[center].extend(path)
#     return total_distance, center_orders, all_paths, pending_orders

# # 模拟一天的配送
# def simulate_day(num_intervals):
#     total_distance = 0
#     all_orders = []  # 存储所有的订单
#     all_paths_per_interval = []  # 用于存储每个时间间隔的所有路径
#     for interval in range(num_intervals):
#         orders_file_n = "./orders/orders_"+ str(interval) + ".txt"
#         new_orders = read_orders_from_file(orders_file_n)
#         all_orders.extend(new_orders)
#         dist,  center_orders, all_paths, pending_orders = decision_system(all_orders, interval * time_interval)
#         total_distance += dist
#         all_paths_per_interval.append((center_orders, all_paths))
#         all_orders = pending_orders  # 更新为未配送的订单
#     return total_distance, all_paths_per_interval

# 订单累积决策系统
def decision_system(orders, accumulated_orders):
    total_distance = 0
    center_orders = defaultdict(list)  # 每个配送中心分配的订单
    for order in orders:
        # 找到距离该订单最近的配送中心
        closest_center = min(distribution_centers, key=lambda center: distance(center, delivery_docks[order[0]]))
        center_orders[closest_center].append(order)

    # 合并累积的订单
    for center in center_orders.keys():
        center_orders[center].extend(accumulated_orders[center])
        accumulated_orders[center] = []

    all_paths = defaultdict(list)  # 用于存储每个配送中心的路径
    for center, orders in center_orders.items():
        # 先处理紧急订单
        urgent_orders = [order for order in orders if priority_levels[order[1]] <= 1.5]
        non_urgent_orders = [order for order in orders if priority_levels[order[1]] > 1.5]

        # 路径规划
        if urgent_orders:
            path, dist = plan_path(center, urgent_orders)
            total_distance += dist
            all_paths[center].extend(path)

        # 累积非紧急订单
        accumulated_orders[center].extend(non_urgent_orders)

        # 如果累积的订单达到一定数量或者某些订单的优先级提高到需要立即处理的程度
        if len(accumulated_orders[center]) >= max_cargo or any(priority_levels[order[1]] <= 1.5 for order in accumulated_orders[center]):
            path, dist = plan_path(center, accumulated_orders[center])
            total_distance += dist
            all_paths[center].extend(path)
            accumulated_orders[center] = []  # 清空累积的订单

    return total_distance, center_orders, all_paths, accumulated_orders

def simulate_day(num_intervals):
    total_distance = 0
    accumulated_orders = defaultdict(list)
    all_paths_per_interval = []  # 用于存储每个时间间隔的所有路径
    for i in range(num_intervals):
        orders_file_n = "./orders/orders_"+ str(interval) + ".txt"
        orders = read_orders_from_file(orders_file_n)  # 从文件读取订单
        interval_distance, center_orders, all_paths, accumulated_orders = decision_system(orders, accumulated_orders)
        total_distance += interval_distance
        all_paths_per_interval.append((center_orders, all_paths))
    return total_distance, all_paths_per_interval

# # 生成配货点和卸货点
# distribution_centers = generate_distribution_centers()

# delivery_docks = generate_delivery_docks(distribution_centers, k, dock_range)

# # 保存生成结果到文件
# with open("./locations.txt", "w") as f:
#     f.write("Distribution Centers:\n")
#     for center in distribution_centers:
#         f.write(f"{center}\n")
#     f.write("\nDelivery Docks:\n")
#     for dock in delivery_docks:
#         f.write(f"{dock}\n")

# 读取完成生成的地图
distribution_centers, delivery_docks = read_locations_from_file()

# 保存位置图
plot_locations(distribution_centers, delivery_docks)

print("生成结果已保存并绘图完成。")

# 生成订单并保存到文件
for i in range(0,48):
    orders = generate_orders()
    file_n = "./orders/orders_"+ str(i) + ".txt"
    save_orders_to_file(orders, file_n)

# 假设一天内有 n 个时间间隔（每间隔 10 分钟，共 10n 分钟）
num_intervals = 24
total_distance, all_paths_per_interval = simulate_day(num_intervals)
print(f"一天的总配送路程: {total_distance:.2f} km")

# 输出每个配送中心要执行的订单及其路径
for interval, (center_orders, all_paths) in enumerate(all_paths_per_interval):
    print(f"\n时间段 {interval + 1}:")
    for center, orders in center_orders.items():
        center_index = distribution_centers.index(center)
        print(f"  配送中心 {center_index}:")
        for order in orders:
            print(f"    订单来自卸货点 {order[0]}, 优先级: {order[1]}")
        print("    路径:")
        for trip_orders, trip_distance in all_paths[center]:
            print(f"      此次配送路程长度 {trip_distance:.2f} km:")
            for order in trip_orders:
                dock_index = order[0]
                print(f"        -> 卸货点 {dock_index}")

