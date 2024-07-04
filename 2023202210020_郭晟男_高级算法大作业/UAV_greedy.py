import random
import numpy as np
import pygame
from deap import base, creator, tools, algorithms
import time

# 生成卸货点函数
def generate_delivery_points(center, num_points_range, distance_px):
    points = []
    for _ in range(random.randint(*num_points_range)):
        while True:
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(0, distance_px)
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            if 0 <= x <= 500 and 0 <= y <= 500:
                points.append((int(x), int(y)))
                break  # 满足条件后退出循环
    return points

# 生成订单函数
def generate_orders(delivery_points, seed, max_orders=3, current_time=0):
    orders = []
    random.seed(seed)
    for i, point in enumerate(delivery_points):
        num_orders = random.randint(0, max_orders)
        for _ in range(num_orders):
            priority = random.choice([0,1,2])# 优先级列表，越小优先级越高
            order = {
                'point': point,
                'point_index':all_locations.index(point),
                'priority': priority,
                'generate_time': current_time,
                'deadline': current_time + priority_time_limits[priority] * 60  # 转换为秒
            }
            orders.append(order)
    return orders

# 计算距离函数
def distance_matrix(locations):
    # 假设的像素到米的转换系数
    pixels_to_meters = 2000/500 # 800px:40km
    size = len(locations)
    dist_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(np.array(locations[i]) - np.array(locations[j])) * pixels_to_meters
    return dist_matrix

def calculate_route_distance(route):
    """计算路径的总距离"""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += dist_matrix[route[i]][route[i + 1]]
    return total_distance

def validate_route(route):
    """验证路径是否满足无人机的最大飞行距离限制"""
    route_distance = calculate_route_distance(route)
    return route_distance <= max_drone_range / 2  # 单程最大飞行距离为20公里，返回后为10公里

def split_orders(orders):
    """将订单按最大载重进行拆分"""
    chunks = [orders[i:i + max_drone_capacity] for i in range(0, len(orders), max_drone_capacity)]
    return chunks

def move_drones(routes):
    positions = [[all_locations[route[0]]] for route in routes]
    targets = [1] * len(routes)
    # 定义初始时间和速度
    start_time = time.time()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        
        # 绘制配送中心和卸货点
        for i, center in enumerate(delivery_centers):
            pygame.draw.circle(screen, (0, 0, 255), center, 5)
            label = pygame.font.Font(None, 24).render(f'C{i}', True, (0, 0, 255))
            screen.blit(label, (center[0]+10, center[1]))
        for i, point in enumerate(delivery_points):
            pygame.draw.circle(screen, (255, 0, 0), point, 5)
            label = pygame.font.Font(None, 24).render(f'P{i}', True, (255, 0, 0))
            screen.blit(label, (point[0]+10, point[1]))
        
        # 绘制路径
        for route in routes:
            for i in range(len(route) - 1):
                start_pos = all_locations[route[i]]
                end_pos = all_locations[route[i + 1]]
                pygame.draw.line(screen, (74, 74, 74), start_pos, end_pos, 1)
        

        # 绘制无人机位置
        drone_speed = 60  # 假设无人机速度是 60 单位/秒
        # 更新无人机位置并绘制
        elapsed_time = time.time() - start_time
        for i, route in enumerate(routes):
            if targets[i] < len(route):
                start_pos = all_locations[route[targets[i] - 1]]
                end_pos = all_locations[route[targets[i]]]
                cur_pos = positions[i][-1]
                vector = np.array(end_pos) - np.array(cur_pos)
                distance = np.linalg.norm(vector)
                
                if distance < drone_speed * elapsed_time:
                    positions[i].append(end_pos)
                    targets[i] += 1
                    start_time = time.time()  # 重置时间
                else:
                    direction = vector / distance
                    new_pos = np.array(cur_pos) + direction * drone_speed * elapsed_time
                    positions[i].append(tuple(new_pos.astype(int)))
                
                pygame.draw.circle(screen, (103,255,2), positions[i][-1], 5)
        # # 绘制无人机位置
        # for i, route in enumerate(routes):
        #     if targets[i] < len(route):
        #         start_pos = all_locations[route[targets[i] - 1]]
        #         end_pos = all_locations[route[targets[i]]]
        #         cur_pos = positions[i][-1]
        #         vector = np.array(end_pos) - np.array(cur_pos)
        #         distance = np.linalg.norm(vector)
                
        #         if distance < drone_speed:
        #             positions[i].append(end_pos)
        #             targets[i] += 1
        #         else:
        #             direction = vector / distance
        #             new_pos = np.array(cur_pos) + direction * drone_speed
        #             positions[i].append(tuple(new_pos.astype(int)))
                
        #         pygame.draw.circle(screen, (0, 255, 0), positions[i][-1], 5)
        pygame.display.flip()
        clock.tick(30)  # 降低帧率以减慢动画速度

        # 添加图例
        font = pygame.font.Font(None, 30)
        legend1 = font.render("delivery center", True, (0, 0, 255))
        legend2 = font.render("unloading point", True, (255, 0, 0))
        legend3 = font.render("UAV path", True, (0, 255, 0))
        screen.blit(legend1, (10, 10))
        screen.blit(legend2, (10, 40))
        screen.blit(legend3, (10, 70))
        
        pygame.display.flip()
        clock.tick(60)

def modified_greedy_algorithm(start_index, order_points, priorities, max_distance):
    unvisited = set(range(len(order_points)))
    # unvisited = {idx: order_points[idx] for idx in range(len(order_points))}
    drones_used = 0
    all_paths = []

    def can_return_to_start(next_index):
        return (dist_matrix[current_index][next_index] + dist_matrix[next_index][start_index]) <= max_distance

    while unvisited:
        path = [start_index]
        current_index = start_index
        total_distance = 0
        distance_next_to_start = 0
        while unvisited:
            # print(1)
            feasible_points = [idx for idx in unvisited if can_return_to_start(order_points[idx])]
            if not feasible_points:
                # print(1)
                break
            # print(2)
            # 选择优先级最高且最近的点
            next_idx = min(feasible_points, key=lambda idx: (priorities[idx], dist_matrix[current_index][order_points[idx]]))    
            next_index = order_points[next_idx]
            distance_to_next = dist_matrix[current_index][next_index]
            distance_next_to_start = dist_matrix[next_index][start_index]
            # print(1)
            if total_distance + distance_to_next + distance_next_to_start > max_distance:
                break

            if(len(path)>max_drone_capacity):
                break
            total_distance += distance_to_next
            path.append(next_index)
            unvisited.remove(next_idx)
            current_index = next_index
            # print("path:",path)
        path.append(start_index)
        total_distance += distance_next_to_start
        # print("center_index:{}------fianl_path:{}".format(start_index, path))
        print(unvisited)
        # print(order_points[3])
        # print(dist_matrix[start_index][order_points[3]])
        
        all_paths.append(path)
        drones_used += 1

    return all_paths, drones_used

def main():
    global total_delivery_time, total_flight_distance, order_response_times, drones_active_time
    start_time = time.time()
    t = 10  # 每隔t分钟生成订单
    time_period = 60  # 模拟时间段，如1小时
    max_orders = 2  # 每个卸货点最多生成订单数量
    max_distance = 20  * 100
    seed = 17
    all_drones = 0
    num_orders = 0

    for iteration in range(0, time_period, t):
        current_time = start_time + iteration * 60
        seed += t
        orders = generate_orders(delivery_points, seed, max_orders, current_time)
        num_orders += len(orders)
        print(f"第{iteration // t + 1}次生成的订单:", orders)
        print(f"第{iteration // t + 1}次生成的订单长度:", len(orders))
        # 将订单分配给最近的配送中心
        center_orders = {i: [] for i in range(len(delivery_centers))}
        for order in orders:
            closest_center = min(range(len(delivery_centers)), key=lambda i: np.linalg.norm(np.array(order['point']) - np.array(delivery_centers[i])))
            center_orders[closest_center].append(order)

        all_routes = []
        drone_ids = 0  # 假设无人机编号从0开始
        step_flight_distance = 0
        step_drones = 0
        for center_index, c_orders in center_orders.items():
            # orders_split = split_orders(c_orders)
            # print(orders_split)
            # for order_chunk in orders_split:
                order_chunk = c_orders
                # 获取当前中心的订单点
                order_points = [order['point'] for order in order_chunk]  # 获取所有当前订单的卸货点
                print(order_points)
                order_points_indices = [all_locations.index(order['point']) for order in order_chunk]
                print(order_points_indices)
                order_points_priority = [order['priority'] for order in order_chunk]
                # local_points = [order['point'] for order in orders if order['point'] in delivery_points and np.linalg.norm(np.array(order['point']) - np.array(center)) <= 200]
                # 应用贪心算法来生成路径
                if order_points:
                    print(f"Center: {center_index}")
                    routes, drones_used = modified_greedy_algorithm(start_index=center_index, order_points=order_points_indices,priorities=order_points_priority,max_distance=max_distance)
                    # print(f"All locations count: {len(all_locations)}")
                    print(f"Best route indices: {routes}")
                    print(f"Drones used for center {center_index}: {drones_used}")
                    step_drones += drones_used
                    # print(f"Best route actual locations: {[all_locations[i] for i in best_route]}")
                    
                    for best_route in routes:
                        all_routes.append(best_route)
                        print(f"配送中心{delivery_centers[center_index]}的无人机路径规划: {best_route}")
                        dists = []
                        for i in range(len(best_route) - 1):
                            dist= dist_matrix[best_route[i]][best_route[i + 1]]
                            dists.append(dist)
                        print(dists)
                        # 计算和记录飞行距离
                        route_distance = calculate_route_distance(best_route)
                        step_flight_distance += route_distance
                    

                    for index in best_route:
                        if index in order_points:
                            order = next((o for o in orders if o['point_index'] == index), None)
                            if order:
                                response_time = time.time() - order['generate_time']
                                order_response_times.append(response_time)

                else:
                    print("No local points or invalid route generated for center", center)

        print(f"第{iteration // t + 1}次总飞行距离:{step_flight_distance / 100} km" )
        print(f"第{iteration // t + 1}次无人机出动总数量:{step_drones} 架次" ) 
        print(f"第{iteration // t + 1}次生成的订单长度:{len(orders)} 单")
        total_flight_distance += step_flight_distance
        all_drones += step_drones
        # 模拟无人机移动
        move_drones(all_routes)

        # 输出性能指标
    print(f"总飞行距离:{total_flight_distance / 100} km" )  # 转换为千米
    print(f"无人机出动总数量:{all_drones} 架次" )
    print(f"订单总数量:{num_orders} 单" )

if __name__ == "__main__":
    # 初始化pygame
    pygame.init()
    screen = pygame.display.set_mode((500, 500))# 对应50 km
    pygame.display.set_caption("UAV path planning")
    clock = pygame.time.Clock()

    random.seed(3047)

    # 随机生成配送中心和卸货点的位置
    num_centers = 5
    # num_points = 10
    delivery_centers = [(random.randint(0, 500), random.randint(0, 500)) for _ in range(num_centers)]
    # delivery_points = [(random.randint(0, 800), random.randint(0, 800)) for _ in range(num_points)]


    # 无人机参数
    drone_speed = 60 * 100 / 3600  # 60 km/h 转换为 十米/秒
    max_drone_capacity = 5  # 无人机一次最多携带5个物品
    max_drone_range = 20 *100  # 无人机最大飞行距离20公里，转换为米

    # 订单优先级与最大配送时间（单位：分钟）
    priority_time_limits = [30, 90, 180]
    # priority_time_limits = {
    #     '一般': 180,  # 3小时
    #     '较紧急': 90,  # 1.5小时
    #     '紧急': 30    # 0.5小时
    # }

    # 初始化性能统计变量
    total_delivery_time = 0
    total_flight_distance = 0
    order_response_times = []
    drones_active_time = {}  # 用于计算每个无人机的活跃时间

    # 每个配送中心生成卸货点
    delivery_points = []
    for center in delivery_centers:
        delivery_points.extend(generate_delivery_points(center, (8, 10), 250))  # 100 像素距离对应10公里

    all_locations = delivery_centers + delivery_points

    dist_matrix = distance_matrix(all_locations)
    main()
    pygame.quit()