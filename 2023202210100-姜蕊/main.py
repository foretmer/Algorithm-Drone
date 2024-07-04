import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque, defaultdict

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号 '-' 显示为方块的问题

# 参数
I = 5  # 配送中心数量
J = 50  # 卸货点数量
max_items_per_drone = 10  # 无人机一次最多携带的物品数量
max_flight_distance = 20  # 无人机一次飞行最远路程
drone_speed = 60  # 无人机速度 (km/h)
order_interval = 60  # 订单生成间隔（分钟）
order_max = 3  # 每个卸货点最大订单数量
simulation_duration = 24 * 60  # 模拟时间（分钟）

# 订单优先级和配送时间要求（分钟）
priority_levels = {
    1: 180,  # 一般
    2: 90,   # 较紧急
    3: 30    # 紧急
}

# 随机种子
np.random.seed(3)
random.seed(3)

# 初始化坐标
xc_depots = np.random.rand(I) * 50
yc_depots = np.random.rand(I) * 50
xc_customers = []
yc_customers = []
while len(xc_customers) < J:
    x = np.random.rand() * 50
    y = np.random.rand() * 50
    distances = [np.sqrt((x - xc_depots[i]) ** 2 + (y - yc_depots[i]) ** 2) for i in range(I)]
    if min(distances) <= max_flight_distance / 3:
        xc_customers.append(x)
        yc_customers.append(y)

xc = np.concatenate((xc_depots, xc_customers))
yc = np.concatenate((yc_depots, yc_customers))

def plot_init(xc, yc):
    # 标记配送中心和卸货点
    for i in range(len(xc)):
        label = f'D{i + 1}' if i < I else f'C{i - I + 1}'
        color = 'red' if i < I else 'blue'  # 配送中心用红色，卸货点用蓝色
        plt.scatter(xc[i], yc[i], marker='D', s=10, color=color)
        plt.text(xc[i], yc[i], label, fontsize=7, color='black', ha='right', va='bottom')

    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.show()

# 绘制初始配送中心和卸货点的位置
plot_init(xc, yc)

# 计算距离矩阵
dist_matrix = np.zeros((I + J, I + J))
for i in range(I + J):
    for j in range(I + J):
        dist_matrix[i, j] = np.sqrt((xc[i] - xc[j]) ** 2 + (yc[i] - yc[j]) ** 2)

# 定义订单类
class Order:
    def __init__(self, point, priority, time):
        self.point = point
        self.priority = priority
        self.time = time

# 计算路径总距离
def fitness(route, depot):
    if not route:
        return 0
    total_distance = dist_matrix[depot, route[0]]
    for i in range(1, len(route)):
        total_distance += dist_matrix[route[i - 1], route[i]]
    total_distance += dist_matrix[route[-1], depot]  # 返回配送中心
    return total_distance

# 生成订单
def generate_orders(current_time):
    orders = deque()
    for j in range(J):
        for _ in range(random.randint(0, order_max)):
            priority = random.choices([1, 2, 3], [0.6, 0.3, 0.1])[0]
            orders.append(Order(point=j, priority=priority, time=current_time))
    return orders

# 检查订单的配送时间要求是否被违反
def is_violated(order, current_time):
    return current_time - order.time > priority_levels[order.priority]

# 贪心算法适应度函数（总距离），考虑无人机携带物品数量的约束
def greedy_algorithm(orders, depot, start_point=None):
    if not orders:
        return []

    route = []
    total_distance = 0
    customer_orders = defaultdict(int)

    for order in orders:
        customer_orders[order.point + I] += 1

    remaining_customers = set(customer_orders.keys())

    if start_point:
        route.append(start_point)
        remaining_customers.discard(start_point)
        total_distance += dist_matrix[depot, start_point]

    while remaining_customers and len(route) < max_items_per_drone:
        if not route:
            next_customer = min(remaining_customers, key=lambda x: dist_matrix[depot, x])
            route.append(next_customer)
            total_distance += dist_matrix[depot, next_customer]
        else:
            last = route[-1]
            next_customer = min(remaining_customers, key=lambda x: dist_matrix[last, x])
            next_distance = dist_matrix[last, next_customer] + dist_matrix[next_customer, depot]
            if total_distance + next_distance <= max_flight_distance:
                route.append(next_customer)
                total_distance += dist_matrix[last, next_customer]
            else:
                break

        customer_orders[next_customer] -= 1
        if customer_orders[next_customer] == 0:
            remaining_customers.remove(next_customer)

    return route

# 绘制最佳路径的函数
def plot_paths(all_paths, xc, yc, W, time_label):
    plt.figure(figsize=(10, 6))

    # 将所有路径合并为一个列表，以便绘图
    flattened_paths = [path for depot_paths in all_paths for path in depot_paths]

    # 绘制配送中心和卸货点
    for i in range(len(xc)):
        label = f'D{i + 1}' if i < I else f'C{i - I + 1}'
        color = 'red' if i < I else 'blue'
        plt.scatter(xc[i], yc[i], marker='D', s=10, color=color)
        if i < I:
            plt.text(xc[i], yc[i], label, fontsize=7, color='red', ha='right', va='bottom')
        else:
            plt.text(xc[i], yc[i], label, fontsize=7, color='black', ha='right', va='bottom')

    # 绘制路径
    for i, path in enumerate(flattened_paths):
        color = plt.cm.viridis(i / len(flattened_paths))
        plt.plot([xc[node] for node in path], [yc[node] for node in path], marker='o', markersize=4, linestyle='-',
                 color=color, label=f'无人机 {i + 1}')

    plt.title(f'Drone Paths at {time_label} minutes')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.show()

# 将路径节点转化为标号
def convert_route_to_labels(route):
    return [f'D{node + 1}' if node < I else f'C{node - I + 1}' for node in route]

# 模拟主循环
global_orders = generate_orders(0)  # 初始化时生成订单
time_intervals = simulation_duration // order_interval

total_distance_all_drones = 0

# 存储每个配送中心的所有路径以便绘图
all_paths = [[] for _ in range(I)]

for t in range(time_intervals):
    current_time = t * order_interval

    # 检查是否有订单违反配送时间要求，并删除它们
    global_orders = [order for order in global_orders if not is_violated(order, current_time)]

    # 更新订单优先级
    for order in global_orders:
        if order.priority < 3:
            new_priority = order.priority + 1
            if current_time - order.time > priority_levels[new_priority]:
                order.priority = new_priority

    # 生成新订单
    new_orders = generate_orders(current_time)
    global_orders.extend(new_orders)

    # 按照优先级排序订单，只处理当前最高优先级的订单
    if t < time_intervals - 1:
        global_orders.sort(key=lambda x: x.priority, reverse=True)
        highest_priority = global_orders[0].priority if global_orders else 1
        highest_priority_orders = [order for order in global_orders if order.priority == highest_priority]
    else:
        highest_priority_orders = global_orders

    # 为每个最高优先级订单找到最优路径
    while highest_priority_orders:
        order = highest_priority_orders[0]
        distances = [dist_matrix[i, order.point + I] for i in range(I)]
        closest_depot = np.argmin(distances)

        # 使用贪心算法找到从该订单出发的最优路径
        if t < time_intervals - 1:
            best_route = greedy_algorithm(highest_priority_orders, closest_depot, start_point=order.point + I)
        else:
            best_route = greedy_algorithm(highest_priority_orders, closest_depot)

        # 将当前最高优先级订单加入路径
        if t < time_intervals - 1:
            best_route = [order.point + I] + [point for point in best_route if point != order.point + I]

        # 检查最佳路径是否超过最大飞行距离
        total_distance = fitness(best_route, closest_depot)
        if total_distance <= max_flight_distance:
            all_paths[closest_depot].append([closest_depot] + best_route + [closest_depot])
            converted_route = convert_route_to_labels(best_route)
            print(f"时间 {current_time} 分钟: 配送中心 D{closest_depot + 1} 的最佳路径: {converted_route}, 距离: {total_distance}")

            # 累加每个无人机的行驶距离到总行驶距离
            total_distance_all_drones += total_distance

            # 派遣无人机根据最佳路径配送订单，并从全局订单中删除这些订单
            for point in best_route:
                customer_point = point - I
                global_orders = [order for order in global_orders if order.point != customer_point]
                highest_priority_orders = [order for order in highest_priority_orders if order.point != customer_point]

    # 输出每个时间段结束时的总行驶距离
    # print(f"时间 {current_time} 分钟后: 总行驶距离: {total_distance_all_drones}")

    # 绘制当前时间段的路径图
    if current_time == 0:
        plot_paths(all_paths, xc, yc, total_distance_all_drones, current_time)

    # 绘制当前时间段的路径图
    if current_time == 1380:
        plot_paths(all_paths, xc, yc, total_distance_all_drones, current_time)

    all_paths = [[] for _ in range(I)]


# 最终输出所有时间段结束时的总行驶距离
print("Total Distance Covered by All Drones: ", total_distance_all_drones)
