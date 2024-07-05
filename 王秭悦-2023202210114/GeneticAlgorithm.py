from util import *

# 常量定义
DELIVERY_RADIUS = 20   # 无人机最大飞行距离 (公里)
DRONE_SPEED = 60       # 无人机速度 (公里/小时)
DRONE_CAPACITY = 3     # 无人机最大载货量
TIME_INTERVAL = 1     # 时间间隔
TIME_LIMITS = {1: 0.5, 2: 1.5, 3: 3.0}  # 订单优先级对应的时间限制 (小时)
DELAY_PENALTY = {1: 10, 2: 5, 3: 1}     # 订单优先级对应的延迟罚分

# 生成初始种群
def generate_initial_population(orders, population_size):
    population = []
    for _ in range(population_size):
        random.shuffle(orders)
        individual = []
        i = 0
        while i < len(orders):
            # 随机确定路径长度
            route_length = random.randint(DRONE_CAPACITY-1, DRONE_CAPACITY)
            route = orders[i:i+route_length]
            individual.append(route)
            i += route_length
        population.append(individual)
    return population

# 适应度函数
def fitness(individual, center, current_time):
    total_distance = 0
    time_violation_penalty = 0
    distance_violation_penalty = 0
    delay_penalty = 0

    for orders in individual:
        if not orders:
            continue

        previous_point = center
        time_elapsed = 0

        for order in orders:
            i, point, priority, order_time = order
            route_distance = calculate_distance(previous_point, point)
            total_distance += route_distance
            if route_distance > DELIVERY_RADIUS:
                distance_violation_penalty += (route_distance - DELIVERY_RADIUS) * 10

            time_elapsed += route_distance / DRONE_SPEED
            time_delay = current_time - order_time
            if time_elapsed+time_delay > TIME_LIMITS[priority]:
                time_violation_penalty += (time_elapsed+time_delay - TIME_LIMITS[priority]) * 10

            previous_point = point
        # 回到配送中心
        route_distance = calculate_distance(previous_point, center)
        total_distance += route_distance
        if route_distance > DELIVERY_RADIUS:
            distance_violation_penalty += (route_distance - DELIVERY_RADIUS) * 10


    weighted_total_distance = (total_distance +
                               time_violation_penalty +
                               distance_violation_penalty +
                               delay_penalty)
    return weighted_total_distance  # 返回加权总距离，最小化该值



# 锦标赛选择
def tournament_selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

# 顺序交叉
def ox(route1, route2):
    size = min(len(route1), len(route2))
    # 选择两个交叉点
    p1, p2 = sorted(random.sample(range(size), 2))
    child1, child2 = [None] * size, [None] * size

    # 拷贝到子代
    child1[p1:p2], child2[p1:p2] = route1[p1:p2], route2[p1:p2]

    # 填补其他剩余位置，保持顺序
    current_pos1, current_pos2 = p2, p2
    for i in range(size):
        if route2[(i + p2) % size] not in child1:
            child1[current_pos1 % size] = route2[(i + p2) % size]
            current_pos1 += 1
        if route1[(i + p2) % size] not in child2:
            child2[current_pos2 % size] = route1[(i + p2) % size]
            current_pos2 += 1

    return child1, child2



def order_crossover(parent1, parent2):
    parent1_merged = merge_lists(parent1)
    parent2_merged = merge_lists(parent2)

    child1_merged, child2_merged = ox(parent1_merged, parent2_merged)
    # print("child1_merged: ", child1_merged)
    # print("child2_merged: ", child2_merged)
    child1 = unmerge_lists(child1_merged, parent1)
    child2 = unmerge_lists(child2_merged, parent2)

    return child1, child2



# 变异操作
def mutate(individual, mutation_rate=0.01):
    new_individual = []
    for route in individual:
        if random.random() < mutation_rate and len(route) > 1:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]

        # 确保每个路径的订单数量不超过4
        while len(route) > DRONE_CAPACITY:
            route.pop()

        new_individual.append(route)

    return new_individual

# 遗传算法主函数
def genetic_algorithm(orders, center, population_size=250, generations=300, mutation_rate=0.01, current_time=0):
    if len(orders)<=1:
        best_individual = [orders]
        best_fitnesses = fitness(best_individual, center, current_time)
    else:
        population = generate_initial_population(orders, population_size)
        # print("初代种群：", population)
        best_fitnesses = []

        for generation in range(generations):
            fitnesses = [fitness(ind, center, current_time) for ind in population]
            best_fitness = min(fitnesses)
            best_fitnesses.append(best_fitness)

            new_population = tournament_selection(population, fitnesses, tournament_size=3)

            child_population = []
            for i in range(0, population_size, 2):
                parent1 = new_population[i]
                parent2 = new_population[i + 1]
                child1, child2 = order_crossover(parent1, parent2)
                mutate(child1, mutation_rate)
                mutate(child2, mutation_rate)
                child_population.extend([child1, child2])

            population = child_population

        best_individual = min(population, key=lambda ind: fitness(ind, center, current_time))

    return best_individual, best_fitnesses

