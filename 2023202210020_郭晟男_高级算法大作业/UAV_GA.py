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
    order_id = 0  # 添加订单唯一标识符
    for i, point in enumerate(delivery_points):
        num_orders = random.randint(0, max_orders)
        for _ in range(num_orders):
            priority = random.choice([0,1,2])# 优先级列表，越小优先级越高
            order = {
                'id': order_id,
                'point': point,
                'point_index':all_locations.index(point),
                'priority': priority,
                'generate_time': current_time,
                'deadline': current_time + priority_time_limits[priority] * 60  # 转换为秒
            }
            orders.append(order)
            order_id += 1  # 增加订单唯一标识符
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
            pygame.draw.circle(screen, (0, 0, 255), center, 8)
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
                
        #         pygame.draw.circle(screen, (171,255,29), positions[i][-1], 5)
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

# 适应度函数
def evalTSP_with_priority(individual, priorities, distance_weight=1.0, priority_weight=100):
    distance = 0
    total_priority = 0
    valid_path = True  # 假设路径初始有效
    # print(f"priorities:{priorities}")
    # print(f"len_priorities:{len(priorities)}")

    # 计算路径上所有连续城市之间的距离
    for i in range(1, len(individual)):
        # print(dist_matrix.shape)
        if dist_matrix[individual[i-1]][individual[i]] == 0:
            valid_path = False  # 如果发现两个连续城市之间的距离为0，标记路径为无效
        distance += dist_matrix[individual[i-1]][individual[i]]
        # print(f"i:{i}")
        total_priority += priorities[i]

    # 添加从最后一个城市回到第一个城市的距离
    if dist_matrix[individual[-1]][individual[0]] == 0:
        valid_path = False
    distance += dist_matrix[individual[-1]][individual[0]]
    total_priority += priorities[0]

    # 如果路径无效，返回一个非常大的数作为惩罚
    if not valid_path or distance == 0:
        return (float('inf'),)  # 使用无穷大作为惩罚值，确保这种路径不会被选择
    
    # 计算加权适应度值，优先级越高，适应度值应越低
    # print(f"total_priority:{total_priority}")
    fitness_value = distance_weight * distance + priority_weight * total_priority

    return (fitness_value,)  # 返回元组形式的适应度
# # 适应度函数
# def evalTSP(individual):
#     distance = 0
#     valid_path = True  # 假设路径初始有效

#     # 计算路径上所有连续城市之间的距离
#     for i in range(1, len(individual)):
#         if dist_matrix[individual[i-1]][individual[i]] == 0:
#             valid_path = False  # 如果发现两个连续城市之间的距离为0，标记路径为无效
#         distance += dist_matrix[individual[i-1]][individual[i]]

#     # 添加从最后一个城市回到第一个城市的距离
#     if dist_matrix[individual[-1]][individual[0]] == 0:
#         valid_path = False
#     distance += dist_matrix[individual[-1]][individual[0]]

#     # 如果路径无效，返回一个非常大的数作为惩罚
#     if not valid_path or distance == 0:
#         return (float('inf'),)  # 使用无穷大作为惩罚值，确保这种路径不会被选择

#     return (distance,)  # 返回元组形式的适应度

# 自定义交叉操作
def custom_crossover(ind1, ind2):
    # 确保两个个体至少有两个元素，才能进行交叉和随机互换操作
    if len(ind1) > 1 and len(ind2) > 1:
        size = min(len(ind1), len(ind2))
        if size > 1:
            # 保留起点，仅交叉其他部分
            cxpoint = random.randint(1, size - 1) # 选择有效的交叉点 
            ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
            
            # # 随机互换部分基因以增加多样性
            # if random.random() > 0.5:
            #     ind1.append(ind2.pop(random.randint(1, len(ind2) - 1)))

    return ind1, ind2

# 自定义变异操作
def custom_mutation(individual, indpb):
    if random.random() < indpb:
        if len(individual) > 1 :
            index = random.randint(1, len(individual) - 1)  # 避免选择起点
            if random.random() > 0.5:  # 变异：随机改变一个城市
                individual[index] = random.choice(range(len(individual)))
            else:  # 变异：随机交换两个城市的位置，除了起点
                index2 = random.randint(1, len(individual) - 1)
                individual[index], individual[index2] = individual[index2], individual[index]
    return (individual,)
# def custom_mutation(individual, indpb):
#     if random.random() < indpb:
#         if random.random() > 0.5 and len(individual) > 1:  # 删除元素
#             individual.pop(random.randint(0, len(individual) - 1))
#         else:  # 添加新元素
#             individual.append(random.choice(range(len(all_locations))))
#     return (individual,)

# 动态定义个体生成函数
def create_individual(order_points_indices_with_center):
    """ 创建基于订单点索引的个体，长度等于订单点数量 """
    return creator.Individual(order_points_indices_with_center)
# def create_individual(order_points_indices_with_center):
#     """ 创建基于订单点索引的个体，长度等于订单点数量 """
#     start_point = order_points_indices_with_center[0]  # 假设起点是第一个位置
#     remaining_points = order_points_indices_with_center[1:]  # 除去起点的其他位置
#     random.shuffle(remaining_points)  # 洗牌操作

#     return creator.Individual([start_point] + remaining_points)

def genetic_algorithm_tsp(order_points, start_index, max_distance, pop_size=100, cxpb=0.7, mutpb=0.2, ngen=200, elite_size=5):
    pop = toolbox.population_custom(n=pop_size) # 创建种群
    hof = tools.HallOfFame(elite_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    drones_used = 0

    for gen in range(ngen):
        # 保留精英个体
        offspring = tools.selBest(pop, elite_size)
        # 选择个体进行交叉和变异
        offspring.extend(toolbox.select(pop, len(pop) - elite_size))
        
        # 进行交叉操作
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        
        # 评估新生成的个体
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        # 替换种群
        pop[:] = offspring

        # 更新精英个体
        hof.update(pop)
        
        # 收集并打印统计数据
        record = stats.compile(pop)
        print(f"Generation {gen}: {record}")
    best_route = hof[0]

    # 分割路径以满足最大距离限制
    all_routes = []
    current_route = [start_index]  # 当前子路径，初始包含起点
    current_distance = 0  # 当前子路径的总距离

    for i in range(1, len(best_route)):
        next_point = best_route[i]  # 下一个点
        distance_to_next = dist_matrix[current_route[-1]][next_point]  # 当前点到下一个点的距离
        distance_back_to_start = dist_matrix[next_point][start_index]  # 下一个点回到起点的距离
        
        # 判断添加下一个点后，当前子路径的总距离是否超过最大续航距离
        if current_distance + distance_to_next + distance_back_to_start <= max_distance:
            current_route.append(next_point)  # 添加下一个点到当前子路径
            current_distance += distance_to_next  # 更新当前子路径的总距离
            if(len(current_route) - 1 > max_drone_capacity):
                current_route.pop() # 去掉新点
                current_distance -= distance_to_next  # 去掉新点
                current_route.append(start_index)  # 当前子路径回到起点
                all_routes.append(current_route)  # 将当前子路径添加到所有子路径中
                current_route = [start_index, next_point]  # 开始新的子路径
                current_distance = dist_matrix[start_index][next_point]  # 重置当前子路径的总距离
        else:
            current_route.append(start_index)  # 当前子路径回到起点
            all_routes.append(current_route)  # 将当前子路径添加到所有子路径中
            current_route = [start_index, next_point]  # 开始新的子路径
            current_distance = dist_matrix[start_index][next_point]  # 重置当前子路径的总距离

    # 确保最后一个子路径回到起点，并添加到所有子路径中
    current_route.append(start_index)
    all_routes.append(current_route)
    drones_used = len(all_routes)
    # algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats=stats, halloffame=hof, verbose=False)
    return all_routes, drones_used
    
def main():
    global total_delivery_time, total_flight_distance, order_response_times, drones_active_time
    start_time = time.time()
    t = 10  # 每隔t分钟生成订单
    time_period = 60  # 模拟时间段，如30分钟
    max_orders = 2  # 每个卸货点最多生成订单数量
    max_distance = 20 * 100  # 无人机最大飞行距离20公里，转换为米
    seed = 17
    all_drones = 0

    for iteration in range(0, time_period, t):
        current_time = start_time + iteration * 60
        seed += t
        orders = generate_orders(delivery_points, seed, max_orders, current_time)
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
            order_chunk = c_orders
            order_points_indices = [all_locations.index(order['point']) for order in order_chunk]
            order_points = {order['id']: order['point'] for order in order_chunk}
            order_points_priority = [order['priority'] for order in order_chunk]
            
            if order_points:
                print(f"Center: {center_index}")
                order_points_indices_with_center = [center_index] + order_points_indices
                order_points_priority_with_center = [0] + order_points_priority
                # 应用遗传算法来生成路径
                toolbox.register("individual_custom", create_individual, order_points_indices_with_center)
                toolbox.register("population_custom", tools.initRepeat, list, toolbox.individual_custom)
                toolbox.register("evaluate", evalTSP_with_priority, priorities=order_points_priority_with_center)
                best_routes, drones_used = genetic_algorithm_tsp(order_points=order_points,start_index=center_index,max_distance=max_distance,pop_size=100, cxpb=0.7, mutpb=0.2, ngen=200, elite_size=5)

                print(f"Best route indices: {best_routes}")
                print(f"Drones used for center {center_index}: {drones_used}")
                step_drones += drones_used
                for best_route in best_routes:
                    # 忽略虚拟节点
                    # best_route = [index for index in best_route if index != dummy_index]

                    # best_route = best_route + [center_index]
                    print(f"Best route indices: {best_route}")
                    print(f"Best route actual locations: {[all_locations[i] for i in best_route]}")

                    all_routes.append(best_route)
                    print(f"配送中心{delivery_centers[center_index]}的无人机路径规划: {best_route}")

                    dists = [dist_matrix[best_route[i]][best_route[i + 1]] for i in range(len(best_route) - 1)]
                    print(dists)

                    # 计算和记录飞行距离
                    route_distance = calculate_route_distance(best_route)
                    step_flight_distance += route_distance

                    for index in best_route:
                        if index in order_points_indices:
                            order = next((o for o in orders if o['point_index'] == index), None)
                            if order:
                                response_time = time.time() - order['generate_time']
                                order_response_times.append(response_time)
            else:
                print("No local points or invalid route generated for center", center_index)

        print(f"第{iteration // t + 1}次总飞行距离:{step_flight_distance / 100} km")
        print(f"第{iteration // t + 1}次无人机出动总数量:{step_drones} 架次" )
        total_flight_distance += step_flight_distance
        all_drones += step_drones
        # 模拟无人机移动
        move_drones(all_routes)

    # 输出性能指标
    print(f"总飞行距离:{total_flight_distance / 100} km")  # 转换为千米
    print(f"无人机出动总数量:{all_drones} 架次" )

if __name__ == "__main__":
    # 初始化pygame
    pygame.init()
    screen = pygame.display.set_mode((500, 500))# 对应50 km
    pygame.display.set_caption("UAV path planning")
    clock = pygame.time.Clock()

    # random.seed(47)
    random.seed(3047)

    # 随机生成配送中心和卸货点的位置
    num_centers = 5
    delivery_centers = [(random.randint(0, 500), random.randint(0, 500)) for _ in range(num_centers)]

    # 无人机参数
    drone_speed = 60 * 100 / 3600  # 60 km/h 转换为 十米/秒
    max_drone_capacity = 5  # 无人机一次最多携带5个物品
    max_drone_range = 20 *100  # 无人机最大飞行距离20公里，转换为米

    # 订单优先级与最大配送时间（单位：分钟）
    priority_time_limits = [30, 90, 180]

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

    # 遗传算法实现路径规划
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(all_locations)), len(all_locations))
    # toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation, indpb=0.05) 
    toolbox.register("select", tools.selTournament, tournsize=3) # 锦标赛规模，即每次选择3个个体进行比较
    # toolbox.register("evaluate", evalTSP)

    main()
    pygame.quit()