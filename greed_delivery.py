import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# 配送中心位置
delivery_center_locations = np.array([[44, 47], [0, 3], [3, 39]])
delivery_center_names = ['Center 1', 'Center 2', 'Center 3']

# 绘制配送中心位置
plt.figure(figsize=(10, 8))
plt.scatter(delivery_center_locations[:, 0], delivery_center_locations[:, 1], c='blue', marker='s', label='Delivery Centers')
for i, (x, y) in enumerate(delivery_center_locations):
    plt.text(x, y+0.5, f'{delivery_center_names[i]} ({x}, {y})', fontsize=12, ha='center', va='bottom')
plt.grid()
plt.show()

# 订单位置和优先级
orders = [((9, 19), 'Very Urgent'), ((21, 36), 'Very Urgent'), ((21, 36), 'Urgent'), ((9, 19), 'Urgent'), ((9, 19), 'Urgent'), 
          ((9, 19), 'Urgent'), ((24, 24), 'Urgent'), ((9, 19), 'General'), ((21, 36), 'General'), ((23, 6), 'General'), 
          ((37, 48), 'Urgent'), ((36, 37), 'General'), ((39, 43), 'Very Urgent'), ((51, 52), 'General'), ((44, 38), 'Urgent')]

# DBSCAN聚类算法进行配送中心和订单的聚类
def cluster_orders(delivery_center_locations, orders, eps=20):
    order_locations = np.array([order[0] for order in orders])
    dbscan = DBSCAN(eps=eps, min_samples=1)
    order_labels = dbscan.fit_predict(order_locations)
    
    center_orders = {i: [] for i in range(len(delivery_center_locations))}
    for label in set(order_labels):
        cluster_orders = order_locations[order_labels == label]
        center_distances = cdist(cluster_orders, delivery_center_locations)
        closest_centers = np.argmin(center_distances, axis=1)
        
        for order_idx, center_idx in enumerate(closest_centers):
            center_orders[center_idx].append(orders[np.where(order_labels == label)[0][order_idx]])
    
    return center_orders

# 聚类结果
center_orders = cluster_orders(delivery_center_locations, orders)
print("\n分配给每个配送中心的订单:")
for center, orders in center_orders.items():
    print(f"配送中心 {center}: {orders}")

def plot_clusters(delivery_center_locations, center_orders, delivery_center_names):
    plt.figure(figsize=(10, 8))
    plt.scatter(delivery_center_locations[:, 0], delivery_center_locations[:, 1], c='blue', marker='s', label='Delivery Centers')
    
    # 提取配送点的位置
    distribution_points = np.array([order[0] for orders in center_orders.values() for order in orders])
    plt.scatter(distribution_points[:, 0], distribution_points[:, 1], c='red', marker='o', label='Distribution Points')
    
    for i, (x, y) in enumerate(delivery_center_locations):
        plt.text(x, y+1, f'{delivery_center_names[i]} ({x}, {y})', fontsize=12, ha='center', va='bottom')
    
    # 仅标记卸货点
    for center, orders in center_orders.items():
        for order in orders:
            if order[1] in ['General', 'Urgent', 'Very Urgent']:
                plt.scatter(order[0][0], order[0][1], color='red')  # 仅标记卸货点，不添加标签
                plt.text(order[0][0] - 1, order[0][1] + 1, f'({order[0][0]}, {order[0][1]})', fontsize=10, ha='right', color='black')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.show()

plot_clusters(delivery_center_locations, center_orders, delivery_center_names)

# 初始化无人机参数
drone_capacity = 2
delivery_range = 50  # 公里，暂时设置更大值来调试
drone_speed = 60  # 公里/小时

# 计算订单优先级
priority_dict = {'Very Urgent': 3, 'Urgent': 2, 'General': 1}

def optimize_delivery_routes(center_orders, delivery_range=50, center_capacity=float('inf')):
    print("\n正在进行路径规划...")
    routes = []
    for center, orders in center_orders.items():
        while orders:
            remaining_capacity = drone_capacity
            current_location = delivery_center_locations[center]
            route = [((tuple(current_location), 'Depart from Center'))]
            part_total_distance = 0  # 初始化总距离
            print(f"\n配送中心 {center} 处理订单，当前位置:", current_location)
            
            # 先处理高优先级订单
            orders.sort(key=lambda x: priority_dict[x[1]], reverse=True)
            
            orders_waiting = []
            while remaining_capacity > 0 and orders:
                closest_order = None
                min_cost = float('inf')
                for order in orders:
                    distance = np.linalg.norm(np.array(order[0]) - current_location)
                    cost = distance / priority_dict[order[1]]
                    if cost < min_cost and distance <= delivery_range:
                        min_cost = cost
                        closest_order = order
                
                if closest_order:
                    route.append((closest_order[0], closest_order[1]))
                    part_total_distance += np.linalg.norm(np.array(closest_order[0]) - current_location)
                    current_location = np.array(closest_order[0])
                    remaining_capacity -= 1
                    orders.remove(closest_order)
                else:
                    # 如果没有找到合适的订单，结束当前路径规划
                    break
            
            route.append(((tuple(delivery_center_locations[center]), 'Return to Center')))
            part_total_distance += np.linalg.norm(current_location - delivery_center_locations[center])
            routes.append((route, part_total_distance))  # 将总距离添加到路径信息中
            print("  返回配送中心:", delivery_center_locations[center])
            # 将等待队列的订单放回到订单列表中，以便下一轮进行处理
            orders.extend(orders_waiting)
            orders_waiting.clear()
            
    print("路径规划完成！")
    return routes

# 执行路径规划和调度
optimized_routes = optimize_delivery_routes(center_orders)

def plot_routes(delivery_center_locations, optimized_routes, delivery_center_names, drone_colors):
    plt.figure(figsize=(10, 8))
    plt.scatter(delivery_center_locations[:, 0], delivery_center_locations[:, 1], c='blue', marker='s', label='Delivery Centers')
    for i, (x, y) in enumerate(delivery_center_locations):
        plt.text(x, y+1, f'{delivery_center_names[i]} ({x}, {y})', fontsize=12, ha='center', va='bottom')
    
    for i, route_info in enumerate(optimized_routes):
        route, part_total_distance = route_info  # 获取路径和总距离信息
        print(f"Route {i+1}: {route}")  # 打印路径信息
        route_points = np.array([location for location, _ in route])  # 提取路径点
        plt.plot(route_points[:, 0], route_points[:, 1], color=drone_colors[i], linestyle='-', marker='o', label=f'Drone {i+1} Route')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title('Drone Delivery Route Planning')
    plt.grid()
    plt.show()

# 输出最终的配送路径
print("\n最终的配送路径:")
for i, (drone_routes, part_total_distance) in enumerate(optimized_routes):
    print(f"无人机 {i+1} 的路径:")
    print(" -> ".join([f"{location} ({priority})" if priority != 'Depart from Center' and priority != 'Return to Center' else f"{location} ({priority})" for location, priority in drone_routes]))

# 指定每个无人机的颜色
drone_colors = ['r', 'g', 'b', 'y', 'c', 'm', 'purple', 'orange', 'pink', 'lime', 'cyan', 'darkred']

# 绘制最终的配送路径，指定无人机颜色
plot_routes(delivery_center_locations, optimized_routes, delivery_center_names, drone_colors)
