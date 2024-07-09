import numpy as np
from generate_locations import generate_map, generate_orders
from merge import aggregate_points
from utils import display_all_path_with_labels, display_aggregate
from GA import GA_path_planning
from collections import defaultdict
from GA2 import genetic_algorithm_vrp

# 参数设置
m = 3  # 每个卸货点生成订单的最大数量
n = 10  # 无人机一次最多携带的物品数量
t = 20  # 时间间隔（分钟）
j = 5  # 配送中心数量
k = 60  # 卸货点数量
map_size = 50  # 地图大小
simulation_duration = 200  # 模拟总时间（分钟） 1440=1day

# 订单优先级和对应的配送时间约束
priority_levels = {
    0: 180,  # 一般：3小时（180分钟）内配送到
    1: 90,  # 较紧急：1.5小时（90分钟）内配送到
    2: 30  # 紧急：0.5小时（30分钟）内配送到
}

# 生成地图
distribution_centers, delivery_points = generate_map(map_size=map_size, num_distribution_centers=j,
                                                     num_delivery_points=k, max_distance=7)
print("\nReturned Distribution Centers Coordinates:")
print(distribution_centers)
print("\nReturned Delivery Points Coordinates:")
print(delivery_points)

# 初始化总订单列表
all_orders = []
all_length_day = 0
# 时间离散化循环
for current_time in range(0, simulation_duration, t):
    print(f"\nCurrent Time: {current_time} minutes")

    # 更新延迟订单的时间约束
    updated_all_orders = []
    for order in all_orders:
        updated_order = (order[0], order[1], order[2] - t)
        updated_all_orders.append(updated_order)
    all_orders = updated_all_orders

    # 生成新订单
    new_orders = generate_orders(k, m)
    print("Generated orders (delivery point, priority, time constraint):")
    print(new_orders)
    all_orders.extend(new_orders)

    # 按照新的延迟订单选择方式进行处理, 一次配送一个卸货点的所有订单
    current_orders = []
    delayed_orders = []
    orders_by_point = defaultdict(list)
    for order in all_orders:
        orders_by_point[order[0]].append(order)
    for point, orders in orders_by_point.items():
        if all(order[2] > 100 for order in orders) and len(orders) < n / 2:
            delayed_orders.extend(orders)
        else:
            if len(orders) < n / 2:
                current_orders.extend(orders)
            else:
                current_orders.extend(orders[:int(n / 2)])
                delayed_orders.extend(orders[int(n / 2):])
    all_orders = delayed_orders

    print("Delayed orders:")
    print(delayed_orders)
    print("Current orders:")
    print(current_orders)

    # 获取有订单的卸货点索引
    active_delivery_points_indices = list(set(order[0] for order in current_orders))

    # 筛选出有订单的卸货点
    active_delivery_points = [delivery_points[i] for i in active_delivery_points_indices]

    # 聚合有订单的客户点
    point_assignments = aggregate_points(distribution_centers, np.array(active_delivery_points), edge_threshold=0.5)

    # 映射回原来的卸货点索引
    full_point_assignments = np.full(len(delivery_points), -1)
    for i, idx in enumerate(active_delivery_points_indices):
        full_point_assignments[idx] = point_assignments[i]
    print("Point assignments (each delivery point's assigned distribution center):")
    print(full_point_assignments)
    display_aggregate(distribution_centers, delivery_points, full_point_assignments, map_size)

    # 对每个配送中心进行路径规划
    total_path_length = []
    paths = []
    path_details = []
    for center_index, center in enumerate(distribution_centers):
        assigned_points_indices = [i for i, x in enumerate(full_point_assignments) if x == center_index]
        if assigned_points_indices:
            assigned_points = np.array([delivery_points[i] for i in assigned_points_indices])
            # 确保 orders 中的卸货点索引与 assigned_points 对应
            assigned_orders = [(assigned_points_indices.index(order[0]), order[1], order[2]) for order in current_orders if
                               order[0] in assigned_points_indices]

            path_distance, planned_path_indices_list = GA_path_planning(center, assigned_points, assigned_orders)
            # path_distance, planned_path_indices_list = genetic_algorithm_vrp(center, assigned_points, assigned_orders)

            total_path_length.append(path_distance)
            for planned_path_indices in planned_path_indices_list:
                planned_path = assigned_points[planned_path_indices]
                planned_path_indices_global = [assigned_points_indices[i] for i in planned_path_indices]
                # 添加返回配送中心的路径
                if len(planned_path) > 0:
                    planned_path = np.vstack((center, planned_path, center))

                # 计算路径的负载和长度
                load = sum([sum(1 for order in assigned_orders if order[0] == i) for i in planned_path_indices])
                path_length = 0
                current_point = center
                for point in planned_path:
                    path_length += np.linalg.norm(current_point - point)
                    current_point = point
                path_length += np.linalg.norm(current_point - center)

                # 记录路径详细信息
                path_details.append({
                    "path": planned_path_indices_global,
                    "load": load,
                    "length": path_length
                })

                paths.append((center, planned_path))

    print(f"Total path length: {total_path_length} = {sum(total_path_length)}")
    all_length_day += sum(total_path_length)

    # 打印每个路径的详细信息
    for detail in path_details:
        path_str = ' -> '.join([str(point) for point in detail['path']])
        print(f"Path: {path_str}", end='\t')
        print(f"Load: {detail['load']}", end='\t')
        print(f"Length: {detail['length']}\n")

    display_all_path_with_labels(distribution_centers, delivery_points, paths, map_size, active_delivery_points_indices)

print(f"Total Length: {all_length_day}")
