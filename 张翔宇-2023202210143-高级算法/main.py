import pygame
import numpy as np
import random
import copy

# 设定参数
map_seed = 42
order_seed = 0
j = 10  # 配送中心数量
k = 60  # 卸货点数量
map_size = 10000  # m  # 地图大小
max_items_per_drone = 5  # 无人机最大带货数量
max_distance_per_trip = 20000  # m
drone_speed = 60000 / 60  # m/min  # 调整无人机速度参数
time_interval = 30  # minutes  # 时间间隔
orders_per_interval = 30  # 每次生成最大订单数量

max_time_hours = 10
max_time = max_time_hours * 60

# 优先级定义
priority_levels = {
    'normal': 180,    # minutes
    'urgent': 90,     # minutes
    'emergency': 30   # minutes
}

class Point:
    def __init__(self, pos):
        self.pos = pos
        self.id = 0
        self.name = ''
    def set_name(self, id, name):
        self.id = id 
        self.name = name

# 生成地图
def generate_points(j, k):
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def is_valid_point(x, y, existing_points, min_distance):
        for pt in existing_points:
            ex, ey = pt.pos
            if distance(x, y, ex, ey) < min_distance:
                return False
        return True
    
    if map_seed is not None:
        np.random.seed(map_seed)
    
    points1 = []
    points2 = []

    while len(points1) < j:
        min_distance = map_size / (int(j**0.5))
        x, y = np.random.uniform(-map_size, map_size, 2)
        if is_valid_point(x, y, points1, min_distance):
            points1.append(Point((int(x), int(y))))
    
    while len(points2) < k:
        min_distance = map_size / (int(k**0.5))
        x, y = np.random.uniform(-map_size, map_size, 2)
        if is_valid_point(x, y, points1, min_distance / 2) and is_valid_point(x, y, points2, min_distance):
            points2.append(Point((int(x), int(y))))
    
    points1.sort(key=lambda p: p.pos[0] + p.pos[1])
    points2.sort(key=lambda p: p.pos[0] + p.pos[1])

    for i in range(len(points1)):
        points1[i].set_name(i, chr(ord("A") + i))
    
    for i in range(len(points2)):
        points2[i].set_name(i, str(i))    

    return points1, points2

# 订单类定义
class Order:
    def __init__(self, time, dp_index, priority):
        self.time = time
        self.dp_index = dp_index
        self.priority = priority
        self.deadline = time + priority_levels[priority]
    def print_order(self, name):
        if hasattr(self, name):
            print(f"{getattr(self, name)}", end=' ')
        else:
            print(f"Invalid attribute name: {name}")


# 生成订单
def generate_orders(time):
    random.seed(time + order_seed)
    num_orders = random.randint(10, orders_per_interval)
    orders = []
    for _ in range(num_orders):
        point = random.randint(0, k-1)
        priority = random.choices(list(priority_levels.keys()), [1, 1, 1])[0]
        orders.append(Order(time, point, priority))
    return orders

dis_matrix_to_upload = [[-1] * k for _ in range(k)]
dis_matrix_to_delivery = [[-1] * k for _ in range(j)]

# 计算距离函数
def calculate_distance(point1, point2):
    global dis_matrix_to_upload, dis_matrix_to_delivery
    if(point1.name < point2.name):
        point1, point2 = point2, point1
    if(point1.name >= 'A'):
        if(dis_matrix_to_delivery[ord(point1.name)-ord("A")][int(point2.name)]==-1):
            dis_matrix_to_delivery[ord(point1.name)-ord("A")][int(point2.name)] = np.linalg.norm(np.array(point1.pos) - np.array(point2.pos))
        return dis_matrix_to_delivery[ord(point1.name)-ord("A")][int(point2.name)]
    else:

        if(dis_matrix_to_upload[int(point2.name)][int(point1.name)]==-1):
            dis_matrix_to_upload[int(point2.name)][int(point1.name)] = np.linalg.norm(np.array(point1.pos) - np.array(point2.pos))
        return dis_matrix_to_upload[int(point2.name)][int(point1.name)]

# 暴力算法
def brute(orders, distribution_centers, unload_points):
    total_distance = 0
    paths = []
    for order in orders:
        # 选择最近的配送中心
        path_tot_dis = 0
        paths_dis=[]
        closest_center = min(distribution_centers, key=lambda center: calculate_distance(center, unload_points[order.dp_index]))
        one_path = [closest_center, unload_points[order.dp_index], closest_center]
        for i in range(len(one_path)-1):
            dis = calculate_distance(one_path[i], one_path[i+1])
            paths_dis.append(dis)
            path_tot_dis += dis
        total_distance += path_tot_dis
        paths.append((one_path, path_tot_dis, paths_dis))  # 加入返回起始点
    return paths, total_distance

def greedy(orders, distribution_centers, unload_points):
    total_distance = 0
    paths = []
    remaining_orders = orders[:]
    
    while remaining_orders:
        path = []
        path_distance = 0
        path_dists = []
        
        # 找到离任意配送中心最近的订单
        min_dis = float('inf')
        closest_order = None
        closest_center = None
        
        for order in remaining_orders:
            for center in distribution_centers:
                distance = calculate_distance(center, unload_points[order.dp_index])
                if distance < min_dis:
                    min_dis = distance
                    closest_order = order
                    closest_center = center
        
        # 初始化路径
        current_pos = closest_center
        path.append(current_pos)
        path.append(unload_points[closest_order.dp_index])
        path_dists.append(min_dis)
        path_distance += min_dis
        remaining_orders.remove(closest_order)
        current_capacity = 1
        current_pos = unload_points[closest_order.dp_index]

        # 寻找下一个最近的订单
        while remaining_orders and current_capacity < max_items_per_drone:
            min_dis = float('inf')
            next_order = None
            
            for order in remaining_orders:
                distance = calculate_distance(current_pos, unload_points[order.dp_index])
                if distance < min_dis:
                    min_dis = distance
                    next_order = order
            
            if next_order:
                fg=False
                next_pos = unload_points[next_order.dp_index]
                for center in distribution_centers:
                    distance = calculate_distance(center, next_pos)
                    if distance < min_dis:
                        fg=True
                        break
                if(fg):
                    break
                # 检查是否超出最大距离
                if path_distance + min_dis + calculate_distance(next_pos, closest_center) <= max_distance_per_trip:
                    path.append(next_pos)
                    path_dists.append(min_dis)
                    path_distance += min_dis
                    current_pos = next_pos
                    current_capacity += 1
                    remaining_orders.remove(next_order)
                else:
                    break
            else:
                break
        
        path.append(closest_center)
        path_dists.append(calculate_distance(current_pos, closest_center))
        path_distance += path_dists[-1]
        
        paths.append((path, path_distance, path_dists))
        total_distance += path_distance
    
    return paths, total_distance

        
# 生成初始种群
def generate_initial_population(orders, size):
    population = []
    for _ in range(size):
        individual = []
        random.shuffle(orders)
        for order in orders:
            individual.append(order.dp_index)
        population.append(individual)
    return population

# 把基因转换为无人机路径
def convert_to_path(individual, distribution_centers, unload_points):
    if(type(unload_points)!=list):
        print(unload_points)
    all_drone_dis = 0
    op = 0
    paths = []
    while(op<len(individual)):
        best_distance = float('inf')
        best_trip_len = 0
        best_start = None
        best_trip_path = []
        best_path_dis = []
        for center in distribution_centers:
            gene_pos = op
            trip_dis = 0
            cur_pos = center
            trip_len=0
            trip_path = [center]
            path_dis = []
            while(gene_pos<len(individual) and ((trip_dis + calculate_distance(cur_pos, unload_points[individual[gene_pos]]) + \
                    calculate_distance(unload_points[individual[gene_pos]], center))\
                        <=max_distance_per_trip) and \
                    (trip_len < max_items_per_drone)):
                path_dis.append(calculate_distance(cur_pos, unload_points[individual[gene_pos]]))
                trip_len += 1
                trip_dis += calculate_distance(cur_pos, unload_points[individual[gene_pos]])
                cur_pos = unload_points[individual[gene_pos]]
                trip_path.append(unload_points[individual[gene_pos]])
                gene_pos+=1
            if best_trip_len < trip_len or (trip_len > 0 and trip_len == best_trip_len and \
                                            trip_dis + calculate_distance(cur_pos, center)< best_distance):
                best_trip_path = trip_path + [center]
                best_trip_len = trip_len
                best_distance = trip_dis + calculate_distance(cur_pos, center)
                best_start = center
                best_path_dis = path_dis + [calculate_distance(cur_pos, center)]
        op += best_trip_len
        all_drone_dis += best_distance
        paths.append((best_trip_path, best_distance, best_path_dis))
    
    return paths, all_drone_dis

# 计算适应度函数
def calculate_fitness(individual, distribution_centers, unload_points):
    _, total_distance = convert_to_path(individual, distribution_centers, unload_points)
    return total_distance


def genetic(orders, distribution_centers, unload_points, pop_size, num_generations):
    mutation_rate = 0.2
    
    # 选择函数
    def selection(population, fitnesses, num_parents):
        selected_indices = np.argsort(fitnesses)[:num_parents]
        return [population[i] for i in selected_indices]
    
    def pmx_crossover(parent1, parent2):
        size = len(parent1)
        
        c1 = random.randint(0, size - 2)
        c2 = random.randint(c1 + 1, size - 1)
        
        prefix1, middle1, suffix1 = parent1[:c1], parent1[c1:c2], parent1[c2:]
        prefix2, middle2, suffix2 = parent2[:c1], parent2[c1:c2], parent2[c2:]
        
        child1 = prefix1 + middle2 + suffix1
        child2 = prefix2 + middle1 + suffix2
        
        def count_numbers(numbers):
            count = {}
            for i in range(size):
                if numbers[i] in count:
                    count[numbers[i]].append(i)
                else:
                    count[numbers[i]] = [i]
            return count
        cnt = count_numbers(parent1)
        cnt_child = count_numbers(child1)
        for i in range(size):
            if(len(cnt_child[child1[i]])>len(cnt[child1[i]])):
                cnt_child[child1[i]].remove(i)
                for jj in range(size):
                    if parent1[jj] not in cnt_child:
                        child1[i] = parent1[jj]
                        cnt_child[parent1[jj]] = [i]
                        break
                    elif len(cnt_child[parent1[jj]]) < len(cnt[parent1[jj]]):
                        child1[i] = parent1[jj]
                        cnt_child[parent1[jj]].append(i)
                        break
        cnt_child = count_numbers(child2)
        for i in range(size):
            if(len(cnt_child[child2[i]])>len(cnt[child2[i]])):
                cnt_child[child2[i]].remove(i)
                for jj in range(size):
                    if parent1[jj] not in cnt_child:
                        child2[i] = parent1[jj]
                        cnt_child[parent1[jj]] = [i]
                        break
                    elif len(cnt_child[parent1[jj]]) < len(cnt[parent1[jj]]):
                        child2[i] = parent1[jj]
                        cnt_child[parent1[jj]].append(i)
                        break
        sorted(child1, key=lambda x: (child1.index(x), x))
        sorted(child2, key=lambda x: (child2.index(x), x))
        return child1, child2

    def mutation(individual, mutation_rate):
        size = len(individual)
        for i in range(size):
            if random.random() < mutation_rate:
                swap_idx = random.randint(0, size - 1)
                # 交换两个位置的值
                individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
        return individual
    
    # print("generating initial population:")
    population = generate_initial_population(orders, pop_size)
    # print("done")
    # print("Start iterating:")
    for generation in range(num_generations):
        fitnesses = [calculate_fitness(individual, distribution_centers, unload_points) for individual in population]
        # if((generation+1)%10==0):
        #     print("generation",generation+1,population[np.argmin(fitnesses)] , "best fitness:", np.min(fitnesses))
        parents = selection(population, fitnesses, pop_size // 2)
        next_population = []
        while len(next_population) < pop_size:
            # print("generate child")
            parent1, parent2 = random.sample(parents, 2)
            # print("pmx_crossover")
            child1, child2 = pmx_crossover(parent1, parent2)

            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            next_population.extend([child1, child2])
        population = next_population

    fitnesses = [calculate_fitness(individual, distribution_centers, unload_points) for individual in population]
    best_individual = population[np.argmin(fitnesses)]
    return convert_to_path(best_individual, distribution_centers, unload_points)

def PSO(orders, distribution_centers, unload_points, pop_size, num_generations):
    w = 0.2 # 惯性因子
    c1 = 0.4 # 自我认知因子
    c2 = 0.4 # 社会认知因子
    cur_best_dis,cur_best_individual =0,[] # 当前最优值、当前最优解，（自我认知部分）
    global_best_dis,global_best_individual = 0,[] # 全局最优值、全局最优解，（社会认知部分）
    population = generate_initial_population(orders, pop_size)
    fitnesses = [calculate_fitness(individual, distribution_centers, unload_points) for individual in population] # 计算种群适应度
    global_best_dis = cur_best_dis = min(fitnesses) # 全局最优值、当前最优值
    global_best_individual = cur_best_individual = population[fitnesses.index(min(fitnesses))] # 全局最优解、当前最优解

    def crossover(individual,cur_best_individual,global_best_individual,w,c1,c2):
        child = [None]*len(individual)
        parent1 = individual
        #轮盘赌操作选择parent2
        randNum = random.uniform(0, sum([w,c1,c2]))
        if randNum <= w:
            parent2 = [individual[i] for i in range(len(individual)-1,-1,-1)]
        elif randNum <= w+c1:
            parent2 = cur_best_individual
        else:
            parent2 = global_best_individual
        
        #parent1-> child
        start_pos = random.randint(0,len(parent1)-1)
        end_pos = random.randint(0,len(parent1)-1)
        if start_pos>end_pos:start_pos,end_pos = end_pos,start_pos
        child[start_pos:end_pos+1] = parent1[start_pos:end_pos+1].copy()
        
        # parent2 -> child
        list1 = list(range(0,start_pos))
        list2 = list(range(end_pos+1,len(parent2)))
        list_index = list1+list2
        j = -1
        for i in list_index:
            for j in range(j+1,len(parent2)+1):
                if child.count(parent2[j]) < parent2.count(parent2[j]):
                    child[i] = parent2[j]
                    break
                        
        return child

    for generation in range(num_generations):
        for i in range(len(population)):
            population[i] = crossover(population[i],cur_best_individual,global_best_individual,w,c1,c2)
            fitnesses[i] = calculate_fitness(population[i], distribution_centers, unload_points)
        
        cur_best_dis,cur_best_individual =  min(fitnesses),population[fitnesses.index(min(fitnesses))]
        if min(fitnesses) <= global_best_dis:
            global_best_dis,global_best_individual =  min(fitnesses),population[fitnesses.index(min(fitnesses))]

    return convert_to_path(global_best_individual, distribution_centers, unload_points)

# Pygame初始化
pygame.init()
screen_size = 800
info_width = 400
screen = pygame.display.set_mode((screen_size + info_width, screen_size))
pygame.display.set_caption("Drone Delivery Simulation")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GREY = (200, 200, 200)

# 坐标转换
def to_screen_coords(point, map_size, screen_size):
    if type(point)==list or type(point)==tuple:
        return (point[0] + map_size) * screen_size // (2 * map_size), (point[1] + map_size) * screen_size // (2 * map_size)
    else:
        return (point.pos[0] + map_size) * screen_size // (2 * map_size), (point.pos[1] + map_size) * screen_size // (2 * map_size)

# 画地图
def draw_map(distribution_centers, unload_points, orders, paths, drones, elapsed_time):
    screen.fill(WHITE)
    
    # 画配送中心和卸货点
    font = pygame.font.SysFont(None, 24)
    for i, center in enumerate(distribution_centers):
        pygame.draw.circle(screen, BLUE, to_screen_coords(center, map_size, screen_size), 5)
        text = font.render(str(center.name), True, BLACK)  # 渲染编号文本
        screen.blit(text, tuple(np.array(to_screen_coords(center, map_size, screen_size)) + np.array([5, -10])))  # 显示编号
    
    for order in orders:
        pygame.draw.circle(screen, RED, to_screen_coords(unload_points[order.dp_index], map_size, screen_size), 10, 2)

    for i, point in enumerate(unload_points):
        pygame.draw.circle(screen, RED, to_screen_coords(point, map_size, screen_size), 5)
        text = font.render(str(point.name), True, BLACK)  # 渲染编号文本
        screen.blit(text, tuple(np.array(to_screen_coords(point, map_size, screen_size)) + np.array([5, -10])))  # 显示编号

    # 画路径
    for path, _, __ in paths:
        for i in range(len(path) - 1):
            pygame.draw.line(screen, BLACK, to_screen_coords(path[i], map_size, screen_size), to_screen_coords(path[i + 1], map_size, screen_size), 1)

    # 画无人机
    for drone in drones:
        pygame.draw.circle(screen, GREEN, to_screen_coords(drone, map_size, screen_size), 3)

# 显示信息
def draw_info(orders, temporary_orders,  paths, current_time):
    font = pygame.font.SysFont(None, 24)
    info_x_start = screen_size + 20
    
    # 画背景
    pygame.draw.rect(screen, GREY, (screen_size, 0, info_width, screen_size))
    
    y_offset = 20
    screen.blit(font.render(f"Current time:{int(current_time)}/min", True, BLACK), (info_x_start, y_offset))
    y_offset += 30
    
    screen.blit(font.render("Orders:", True, BLACK), (info_x_start, y_offset))
    y_offset += 30
    sorted_orders = copy.deepcopy(orders)
    sorted_orders.sort(key=lambda order:order.dp_index)
    # 显示订单信息
    fg=0
    for i, order in enumerate(sorted_orders):
        text = f"[{order.dp_index}]: ({order.deadline})"
        screen.blit(font.render(text, True, BLACK), (info_x_start, y_offset))
        if fg==0:
            info_x_start += 150
            fg+=1
        elif fg==1:
            info_x_start += 150
            fg+=1
        else:
            y_offset += 25
            info_x_start -= 300
            fg = 0
    if fg!=0:
        y_offset += 25
        info_x_start -= fg*150
    y_offset += 20

    screen.blit(font.render("Temporary Orders:", True, BLACK), (info_x_start, y_offset))
    y_offset += 30
    sorted_orders = copy.deepcopy(temporary_orders)
    sorted_orders.sort(key=lambda order:order.dp_index)
    # 显示订单信息
    fg=0
    for i, order in enumerate(sorted_orders):
        text = f"[{order.dp_index}]: ({order.deadline})"
        screen.blit(font.render(text, True, BLACK), (info_x_start, y_offset))
        if fg==0:
            info_x_start += 150
            fg+=1
        elif fg==1:
            info_x_start += 150
            fg+=1
        else:
            y_offset += 25
            info_x_start -= 300
            fg = 0
    if fg!=0:
        y_offset += 25
        info_x_start -= fg*150
    y_offset += 20

    screen.blit(font.render("Drones:", True, BLACK), (info_x_start, y_offset))
    y_offset += 30
    
    # # 显示路径信息
    for i, (path, _, __) in enumerate(paths):
        text = f"Drone {i+1}: {path[0].name}"
        for jj in range(1, len(path)):
            text += f" -> {path[jj].name}"
        screen.blit(font.render(text, True, BLACK), (info_x_start, y_offset))
        y_offset += 25
def store_and_load(all_orders, distribution_centers, unload_points, current_time):
    all_orders.sort(key=lambda order: order.deadline)
    order_min_center=[]
    for order in all_orders:
        min_dis = float('inf')
        for center in distribution_centers:
            distance = calculate_distance(center, unload_points[order.dp_index])
            if distance < min_dis:
                min_dis = distance
        order_min_center.append(min_dis)

    for i in range(len(all_orders)):
        for jj in range(i+1, len(all_orders)):
            distance = calculate_distance(unload_points[all_orders[i].dp_index], unload_points[all_orders[j].dp_index])
            if(distance <= order_min_center[j]):
                all_orders[j].deadline = all_orders[i].deadline
    orders = []
    temporary_orders = []
    for order in all_orders:
        if(order.deadline<=(current_time+30)):
            orders.append(order)
        else:
            temporary_orders.append(order)
    return orders, temporary_orders
# 主函数
if __name__ == "__main__":
    distribution_centers, unload_points = generate_points(j, k)
    current_time = 0
    orders = []
    temporary_orders = []
    exit_flag = True
    all_time_orders = 0
    all_time_drones = 0
    all_time_dis = 0
    using_temporary_order = True # False,  True
    algorithm = 'greedy' # brute, greedy, genetic, PSO
    pop_size = 50
    num_generations = 200
    while current_time <= max_time:
        orders = generate_orders(current_time)
        if using_temporary_order:
            if(current_time == max_time):
                orders = orders + temporary_orders
                temporary_orders = []
            else:
                orders, temporary_orders = store_and_load(orders + temporary_orders, distribution_centers, unload_points, current_time)
        if orders:
            if(algorithm == 'brute'):
                paths, total_distance = brute(orders, distribution_centers, unload_points)
            if(algorithm == 'greedy'):
                paths, total_distance = greedy(orders, distribution_centers, unload_points)
            if(algorithm == 'genetic'):
                paths, total_distance = genetic(orders, distribution_centers, unload_points, pop_size, num_generations)
            if(algorithm == 'PSO'):
                paths, total_distance = PSO(orders, distribution_centers, unload_points, pop_size, num_generations)
        else:
            paths=[]
            total_distance=0
        one_time_order = len(orders)
        one_time_drones = len(paths)
        if using_temporary_order:
            temporary_orders_len = len(temporary_orders)
            print("订单数：",one_time_order,"暂存订单数：",temporary_orders_len,"无人机架次：",one_time_drones,"总距离：",total_distance)
        else:
            print("订单数：",one_time_order,"无人机架次：",one_time_drones,"总距离：",total_distance)

        all_time_orders += one_time_order
        all_time_drones += one_time_drones
        all_time_dis += total_distance

        drones = [[path[0].pos[0], path[0].pos[1]] for path, _, __ in paths]
        running = True
        clock = pygame.time.Clock()
        elapsed_time = 0
        paused = False
        cur_pos_idx = [0]*len(paths)
        while running:
            if(elapsed_time >= time_interval):
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    exit_flag = False
                elif event.type == pygame.KEYDOWN:
                    # 检查是否按下了空格键
                    if event.key == pygame.K_SPACE:
                        # 切换暂停状态
                        paused = not paused
            if(paused):
                continue
            elapsed_time += clock.get_time() / 1200 # 200,1200 # 加快时间步长
            draw_map(distribution_centers, unload_points,orders, paths, drones, elapsed_time)
            draw_info(orders, temporary_orders, paths, elapsed_time + current_time)

            # 更新无人机位置并绘制路径
            for i in range(len(paths)):
                path, path_tot_dis, paths_dis = paths[i]
                travel_time = sum(paths_dis[:(cur_pos_idx[i]+1)]) / drone_speed
                tot_time = path_tot_dis / drone_speed
                if elapsed_time < tot_time:
                    if(elapsed_time >= travel_time):
                        cur_pos_idx[i]+=1
                        travel_time = sum(paths_dis[:(cur_pos_idx[i]+1)]) / drone_speed
                    last_time = sum(paths_dis[:(cur_pos_idx[i])]) / drone_speed
                    if(travel_time-last_time<=1e-9):
                        ratio=0
                    else:
                        ratio = (elapsed_time-last_time) / (travel_time-last_time)
                    cur_pos = path[cur_pos_idx[i]].pos
                    cur_to = path[cur_pos_idx[i]+1].pos

                    drone_x = cur_pos[0] + (cur_to[0] - cur_pos[0]) * ratio
                    drone_y = cur_pos[1] + (cur_to[1] - cur_pos[1]) * ratio

                    pygame.draw.circle(screen, GREEN, to_screen_coords((drone_x, drone_y), map_size, screen_size), 3)
                # else:
                #     pygame.draw.circle(screen, GREEN, to_screen_coords(return_start, map_size, screen_size), 3)

            pygame.display.flip()
            clock.tick(60)  # 控制帧率

        current_time += time_interval
        drones = []

        if not exit_flag:
            break
        # pygame.quit()
        # sys.exit()
    print("10小时总订单数：",all_time_orders,"总无人机架次：",all_time_drones,"总距离：",all_time_dis)
    print(algorithm, using_temporary_order)
    if(algorithm =='genetic'):
        print("pop_size:",pop_size, "num_generations:", num_generations)
    print(f"map seed:{map_seed}, order seed:{order_seed}")