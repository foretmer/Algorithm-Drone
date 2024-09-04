import random
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import heapq
# 假设的一些常量
MAX_LOAD = 10  # 无人机最大携带物品数量
MAX_DISTANCE = 20  # 最大往返距离
n_centers = 5
SPEED = 60  # 速度为60公里/小时
DELIVERY_TIME = {'普通': 180, '较紧急': 90, '紧急': 30}  # 最大配送时间，单位分钟
DISTANCE_BETWEEN_POINTS = 10  # 配送中心与卸货点之间的最大距离

# 设置随机种子以保证可复现性
random.seed(41)

# 随机生成n个配送中心的坐标
def generate_centers_and_drops(n):
    centers_coords = {f'Center{i+1}': (random.uniform(0, 100), random.uniform(0, 100)) for i in range(n)}
    drop_points_coords = {}
    drop_points = []

    for center, (cx, cy) in centers_coords.items():
        # 每个center随机分配3-7个drop
        mi = random.randint(5, 10)

        for i in range(mi):
            drop_name = f'Drop{len(drop_points) + 1}'
            drop_points.append(drop_name)
            # 生成距离中心点的随机距离和随机角度
            distance = random.uniform(0, DISTANCE_BETWEEN_POINTS)
            angle = random.uniform(0, 2 * math.pi)
            # 根据距离和角度计算新点的坐标
            drop_x = cx + distance * math.cos(angle)
            drop_y = cy + distance * math.sin(angle)
            drop_points_coords[drop_name] = (drop_x, drop_y)

    return centers_coords, drop_points_coords, drop_points

# 生成随机的距离矩阵，包括配送中心到卸货点和卸货点之间的距离
def calculate_distances(centers_coords, drop_points_coords):
    distances = {}
    all_points_coords = {**centers_coords, **drop_points_coords}
    points = list(all_points_coords.keys())
    
    for point1 in points:
        for point2 in points:
            if point1 == point2:
                distances[(point1, point2)] = 0
            else:
                x1, y1 = all_points_coords[point1]
                x2, y2 = all_points_coords[point2]
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                distances[(point1, point2)] = distance
    
    return distances


# m_drops = 15
centers_coords, drop_points_coords, drop_points = generate_centers_and_drops(n_centers)

centers = list(centers_coords.keys())



distances = calculate_distances(centers_coords, drop_points_coords)





# 打印距离矩阵
def print_distance_matrix():
    print("距离矩阵：")
    points = list(centers_coords.keys()) + list(drop_points_coords.keys())
    header = "      " + " ".join(f"{point:8}" for point in points)
    print(header)
    for point1 in points:
        row = f"{point1:6}"
        for point2 in points:
            distance = distances.get((point1, point2), float('inf'))
            row += f" {distance:8.2f}"
        print(row)
    print()

# 生成订单
def generate_orders(time_interval):
    orders = []
    print(drop_points)
    for drop in drop_points:
        num_orders = random.randint(0, 5)  # 每个点生成0到3个订单
        for _ in range(num_orders):
            priority = random.choice(list(DELIVERY_TIME.keys()))
            orders.append((drop, priority))
    return orders

# 将订单分配给最近的配送中心
def assign_orders_to_centers(orders):
    center_orders = defaultdict(list)
    for order in orders:
        drop, priority = order
        # 找到离卸货点最近的配送中心
        closest_center = min(centers, key=lambda center: distances[(center, drop)])
        center_orders[closest_center].append(order)
    return center_orders

# 从配送中心分配订单给无人机
import heapq

def assign_orders_to_drones(center_orders, distances, MAX_LOAD, MAX_DISTANCE):
    
    all_paths = []
    
    for center, orders_list in center_orders.items():
        # 使用堆来管理订单，基于与配送中心的距离排序
        orders_heap = []
        for order in orders_list:
            heapq.heappush(orders_heap, (distances[(center, order[0])], order))
        
        drone_path = [center]
        total_distance = 0
        num_orders = 0

        while orders_heap:
            # 检查堆顶订单但不从堆中移除
            _, next_order = orders_heap[0]
            drop, priority = next_order
            
            if len(drone_path) > 1:
                last_drop = drone_path[-1]
                trip_distance = distances[(last_drop, drop)]
                dist_to_center = distances[(center, drop)]

                # 如果到配送中心的距离小于到上一个点的距离，则结束当前无人机的订单处理
                if dist_to_center < trip_distance:
                    # 完成当前无人机路径并打印
                    drone_path.append(center)
                    print(f"无人机从{center}出发的路径: {' -> '.join(drone_path)}，总距离: {total_distance + distances[(last_drop, center)]:.2f}公里")
                    all_paths.append(drone_path)
            
                    # 重置并开始新的无人机路径配置
                    drone_path = [center]
                    total_distance = 0
                    num_orders = 0
                    continue  # 继续处理剩余的订单

            trip_distance = distances[(center, drop)] if len(drone_path) == 1 else distances[(drone_path[-1], drop)]

            # 考虑返回到配送中心的距离
            return_distance = distances[(drop, center)]
            potential_total_distance = total_distance + trip_distance + return_distance
            
            # 检查距离、负载和时间限制
            if num_orders < MAX_LOAD and potential_total_distance <= MAX_DISTANCE:
                # 计算完成当前订单需要的时间
                delivery_time_required = trip_distance / SPEED * 60  # 转换为分钟
                if delivery_time_required <= DELIVERY_TIME[priority]:
                    # 订单符合要求，将其从堆中正式移除
                    heapq.heappop(orders_heap)
                    drone_path.append(drop)
                    total_distance += trip_distance
                    num_orders += 1
                else:
                    # 当前订单无法在优先级时间内完成，跳过该订单
                    break
            else:
                # 达到负载或距离上限，无人机返回中心
                drone_path.append(center)
                print(f"无人机从{center}出发的路径: {' -> '.join(drone_path)}，总距离: {total_distance + distances[(last_drop, center)]:.2f}公里")
                all_paths.append(drone_path)
                
                # 重置并开始新的路径配置
                drone_path = [center]
                total_distance = 0
                num_orders = 0

        # 确保最后一个路径闭合返回到配送中心
        if drone_path[-1] != center:
            drone_path.append(center)
            print(f"无人机从{center}出发的路径: {' -> '.join(drone_path)}，总距离: {total_distance + distances[(drone_path[-2], center)]:.2f}公里")
            all_paths.append(drone_path)

    return all_paths

def calculate_total_distance(all_paths, distances):
    total_distance = 0
    for path in all_paths:
        # 计算单个无人机的往返总距离
        path_distance = 0
        for i in range(len(path) - 1):
            path_distance += distances[(path[i], path[i+1])]
        total_distance += path_distance
    return total_distance


# 绘制所有路径
def plot_all_paths(all_paths):
    # 绘制所有路径
    colors = plt.get_cmap('tab10')

    plt.figure()
    
    # 绘制所有路径
    for idx, path in enumerate(all_paths):
        path_coords = [centers_coords.get(point, drop_points_coords.get(point)) for point in path]
        plt.plot(*zip(*path_coords), marker='o', color=colors(idx % 10), label=f'route {idx+1}')
        for point in path:
            coord = centers_coords.get(point, drop_points_coords.get(point))
            plt.annotate(point, coord)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('route')
    plt.legend()
    plt.show()







# 主函数
def main():
    print_distance_matrix()  # 打印距离矩阵
    orders = generate_orders(5)  # 假设每5分钟生成一次订单
    print(orders)
    print('订单数：',len(orders))

    center_orders = assign_orders_to_centers(orders)
    all_paths=assign_orders_to_drones(center_orders,distances,MAX_LOAD,MAX_DISTANCE)
    total_flying_distance = calculate_total_distance(all_paths,distances)
    print('总飞行里程：',total_flying_distance,'公里')
    plot_all_paths(all_paths)

if __name__ == "__main__":
    main()
