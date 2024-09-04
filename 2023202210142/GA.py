import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# 参数设置
MAX_DISTANCE = 20  # 无人机一次飞行最远路程（包括回程）
MAX_ITEMS = 10  # 无人机一次最多携带的物品数量

POPULATION_SIZE = 100  # 初始化种群大小
GENERATIONS = 200  # 迭代代数
MUTATION_RATE = 0.1  # 变异率
CROSSOVER_RATE = 0.8  # 交叉率


# 编码函数
def encode(delivery_points):
    return np.random.permutation(len(delivery_points))


# 解码函数
def decode(center, delivery_points, individual, order_count):
    routes = []
    current_route = []
    current_load = 0
    current_distance = 0.0
    current_point = center

    for index in individual:
        next_point = delivery_points[index]
        distance_to_next = np.linalg.norm(current_point - next_point)
        distance_back_to_center = np.linalg.norm(next_point - center)

        if (current_load + order_count[index] <= MAX_ITEMS and
                current_distance + distance_to_next + distance_back_to_center <= MAX_DISTANCE):
            current_route.append(index)
            current_load += order_count[index]
            current_distance += distance_to_next
            current_point = next_point
        else:
            routes.append(current_route)
            current_route = [index]
            current_load = order_count[index]
            current_distance = np.linalg.norm(center - next_point)
            current_point = next_point

    if current_route:
        routes.append(current_route)

    return routes


# 评估函数
def evaluate(individual, center, delivery_points, order_count):
    # center = np.array(center)
    total_distance = 0.0
    routes = decode(center, delivery_points, individual, order_count)

    for route in routes:
        current_point = center
        for index in route:
            next_point = delivery_points[index]
            total_distance += np.linalg.norm(current_point - next_point)
            current_point = next_point
        total_distance += np.linalg.norm(current_point - center)

    return (total_distance,)


# 顺序交叉函数
def order_crossover(ind1, ind2):
    size = len(ind1)
    start, end = sorted(np.random.randint(0, size, 2))

    child1 = creator.Individual(ind1.copy())
    child2 = creator.Individual(ind2.copy())

    # Apply crossover between start and end
    child1[start:end + 1], child2[start:end + 1] = ind2[start:end + 1], ind1[start:end + 1]

    def fill_remaining(child, parent):
        pos = end + 1
        parent_pos = end + 1

        while pos != start:
            if parent_pos == size:
                parent_pos = 0
            if pos == size:
                pos = 0

            if parent[parent_pos] not in child[start:end + 1]:
                child[pos] = parent[parent_pos]
                pos += 1

            parent_pos += 1

    fill_remaining(child1, ind1)
    fill_remaining(child2, ind2)

    return child1, child2


# 逆排序突变函数
def inversion_mutation(individual, indpb):
    if np.random.rand() < indpb:
        size = len(individual)
        start, end = sorted(np.random.randint(0, size, 2))
        individual[start:end + 1] = individual[start:end + 1][::-1]
    return individual,


# 锦标赛选择函数
def tournament_selection(population, k, tournsize):
    return tools.selTournament(population, k, tournsize=tournsize)


def GA_path_planning(center, delivery_points, orders, population_size=POPULATION_SIZE, generations=GENERATIONS,
                     mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE):
    num_points = len(delivery_points)

    # 计算每个卸货点的订单数量
    order_count = [sum(1 for order in orders if order[0] == i) for i in range(num_points)]

    # 定义适应度和个体
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # 初始化工具箱
    toolbox.register("indices", encode, delivery_points)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册工具
    toolbox.register("mate", order_crossover)
    toolbox.register("mutate", inversion_mutation, indpb=mutation_rate)
    toolbox.register("select", tournament_selection, tournsize=3)
    toolbox.register("evaluate", evaluate, center=center, delivery_points=delivery_points, order_count=order_count)

    # 初始化种群
    population = toolbox.population(n=population_size)

    # 评估初始种群的适应度
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # 统计数据
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 记录每一代的最小适应度值
    min_fitness_values = []

    # 使用algorithms.eaMuPlusLambda运行遗传算法
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=population_size,
                                                    lambda_=population_size, cxpb=crossover_rate,
                                                    mutpb=mutation_rate, ngen=generations, stats=stats, verbose=True)

    # 记录每一代的最小适应度值
    min_fitness_values.extend(logbook.select("min"))

    # 绘制适应度函数最小值的变化图
    # plt.plot(min_fitness_values)
    # plt.title("Minimum Fitness over Generations")
    # plt.xlabel("Generation")
    # plt.ylabel("Minimum Fitness")
    # plt.grid(True)
    # plt.show()

    # 获取最优个体及其路径距离
    best_individual = tools.selBest(population, k=1)[0]
    total_distance = best_individual.fitness.values[0]

    # 将最优个体路径索引转换为多个路径索引列表
    planned_path_indices_list = decode(center, delivery_points, best_individual, order_count)

    return total_distance, planned_path_indices_list
