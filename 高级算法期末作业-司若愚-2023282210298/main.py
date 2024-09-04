import numpy as np
import random

bar_count = 80
# 常量定义
J = 3  # 配送中心数量
K = J * 3  # 卸货点数量
N = 6  # 每次最多携带物品
T = 10  # 时间间隔（分钟）
P = [180, 90, 30]  # 优先级
PW = [0.1, 0.2, 0.7]  # 优先级权重
W = 10  # 送货区域大小（公里）
MAX_DISTANCE = 20  # 每次飞行最远距离（公里）
MAX_ORDER_NUM = N  # 每次最大订单生成数
TOTAL_TIME = 60  # 总时间（分钟）
TOTAL_DISTANCE = 0  # 总距离
ORDERS = []  # 订单列表
ORDER_ID_COUNTER = 0  # 订单ID计数器
DRONE_ID_COUNTER = 0  # 无人机ID计数器
DELIVERED = []  # 规避池
CHOICES = []  # 决策池
LOG = []  # 决策记录


def genPos(n, exclude=None):
    """生成n个不重复的坐标点"""
    ps = set()
    while len(ps) < n:
        item = (random.randint(0, W), random.randint(0, W))
        if not (exclude and item in exclude):
            ps.add(item)
    return list(ps)


# 生成送货中心和卸货点坐标
def generate_locations(num_centers, num_droppoints):
    # 生成不包含重复元素的centers坐标
    centers = genPos(J)
    droppoints = genPos(K, exclude=centers)
    return centers, droppoints


# 随机生成订单
def generate_orders(droppoints, max_orders):
    global ORDER_ID_COUNTER
    orders = []
    for dp in droppoints:
        num_orders = random.randint(0, max_orders)
        for _ in range(num_orders):
            priority = random.choices(P, PW)[0]
            ORDER_ID_COUNTER += 1
            orders.append({'id': ORDER_ID_COUNTER, 'location': dp, 'priority': priority})
    # 按优先级排序订单
    return sorted(orders, key=lambda x: x['priority'])


# 计算距离矩阵
def create_distance_matrix(locations):
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
    return distance_matrix


# 适应度函数
def fitness(individual, distance_matrix, max_carry, max_distance):
    total_distance = 0
    segment_distance = 0
    carried_items = 0
    for i in range(1, len(individual)):
        carried_items += 1
        segment_distance += distance_matrix[individual[i - 1]][individual[i]]

        if carried_items > max_carry or segment_distance > max_distance:
            return 1e6  # 超出携带量或最大飞行距离，适应度值设为无穷大
        else:
            total_distance += distance_matrix[individual[i - 1]][individual[i]]

    return total_distance


# 遗传算法
def genetic_algorithm(num_orders, distance_matrix, max_carry, max_distance, pop_size=100, generations=200,
                      mutation_rate=0.01):
    # 初始化种群
    population = init_population(pop_size, num_orders)
    best_individual = None
    best_fitness = float('inf')
    # while best_fitness > max_distance:
    # 计算初始种群中的最佳个体和其适应度值
    best_individual = min(population, key=lambda ind: fitness(ind, distance_matrix, max_carry, max_distance))
    best_fitness = fitness(best_individual, distance_matrix, max_carry, max_distance)

    # 迭代遗传算法
    for _ in range(generations):
        # 选择操作
        population = selection(population, distance_matrix, max_carry, max_distance)
        next_population = []

        # 交叉和变异操作
        pop_size = len(population)  # 重新计算选择后的种群大小
        if pop_size % 2 != 0:  # 确保种群大小为偶数
            population.append(population[0])
            pop_size += 1

        for i in range(0, pop_size, 2):
            parent1, parent2 = population[i], population[i + 1]
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            next_population.extend([child1, child2])

        # 更新种群
        population = next_population

        # 计算当前种群中的最佳个体和其适应度值
        current_best = min(population, key=lambda ind: fitness(ind, distance_matrix, max_carry, max_distance))
        current_fitness = fitness(current_best, distance_matrix, max_carry, max_distance)

        # 更新最佳个体和其适应度值
        if current_fitness < best_fitness:
            best_individual, best_fitness = current_best, current_fitness

    return best_individual, best_fitness


# 初始化种群
def init_population(pop_size, num_orders):
    return [np.random.permutation(num_orders).tolist() for _ in range(pop_size)]


# 选择操作
def selection(population, distance_matrix, max_carry, max_distance):
    population_fitness = [(ind, fitness(ind, distance_matrix, max_carry, max_distance)) for ind in population]
    population_fitness.sort(key=lambda x: x[1])
    return [ind for ind, _ in population_fitness[:len(population) // 2]]


# 交叉操作
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    pointer = 0
    for item in parent2:
        if item not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = item
    return child


# 变异操作
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            swap_idx = np.random.randint(0, len(individual))
            individual[i], individual[swap_idx] = individual[swap_idx], individual[i]


# 包含订单数量、路径、以及距离的OrderDrone类
class Drone:
    def __init__(self, id, orders, route, distance):
        self.id = id
        self.orders = orders
        self.route = route
        self.distance = distance


# 合并两条路径
def merge_drone(d1, d2, dis):
    global DRONE_ID_COUNTER
    # 如果两个无人机的订单数相加不超过N，且总距离不超过MAX_DISTANCE，则合并
    orders = d1.orders + d2.orders
    half = (d1.distance + d2.distance) / 2
    new_dis = half + dis
    if len(orders) <= N and new_dis <= MAX_DISTANCE and dis < half:
        DRONE_ID_COUNTER += 1
        return Drone(DRONE_ID_COUNTER, orders, d1.route + d2.route, new_dis)
    else:
        return None


# 合并路径集合
def merge_drones(drones, all_locations, distance_matrix):
    # 如果drones数量不小于2，找出其中订单数最少的两个尝试合并
    while len(drones) >= 2:
        min1 = min(drones, key=lambda x: len(x.orders))
        idx1 = all_locations.index(min1.route[-1])
        drones.remove(min1)
        min2 = min(drones, key=lambda x: len(x.orders))
        idx2 = all_locations.index(min2.route[0])
        drones.remove(min2)
        merged = merge_drone(min1, min2, distance_matrix[idx1][idx2])
        if merged:
            drones.append(merged)
        else:
            drones.append(min1)
            drones.append(min2)
            break


def print_drones(drones, center=None):
    # 输出无人机负责的订单和路径
    total_dis = 0
    for drone in drones:
        print(f"    无人机[{drone.id}]派送订单：", end="")
        print(f"{[order['id'] for order in drone.orders]}, ", end="")
        print(f"路径：{[center] + drone.route + [center]}, 路程：{drone.distance:.2f} km")
        total_dis += drone.distance
    print(f"    共计：{total_dis:.2f} km")


# 添加路径选择
def add_choices(drones, center):
    global CHOICES
    for drone in drones:
        CHOICES.append((drone.id, [order['id'] for order in drone.orders], [center] + drone.route + [center],
                        drone.distance.__round__(2), center))

        # 路径规划


def route_planning():
    global ORDERS, DRONE_ID_COUNTER, CHOICES, DELIVERED, TOTAL_DISTANCE

    centers, droppoints = generate_locations(J, K)  # 生成送货中心和卸货点
    print("-" * bar_count)
    print("【随机生成坐标】")
    print("配送中心坐标：", centers)
    print("卸货点坐标：", droppoints)

    for current_time in range(0, TOTAL_TIME, T):
        total_dis = 0
        print('-' * bar_count)
        # 更新订单优先级
        for order in ORDERS:
            order['priority'] -= T
        # 生成新订单
        new_orders = generate_orders(droppoints, MAX_ORDER_NUM)
        print(f"【当前周期（分钟）：{current_time}~{current_time + T}，生成订单数量：{len(new_orders)}】")

        ORDERS.extend(new_orders)  # 将新订单添加到订单列表
        # 重新按优先级排序
        ORDERS = sorted(ORDERS, key=lambda x: x['priority'])
        # 当前订单情况
        print(f"   *更新订单详情（按优先级）：{len(ORDERS)}个")
        for order in ORDERS:
            print(
                f"    订单[{order['id']}]：卸货点{order['location']}, 优先级{order['priority']}")

        order_locations = [order['location'] for order in ORDERS]  # 订单的卸货点坐标
        all_locations = centers + order_locations  # 所有坐标
        distance_matrix = create_distance_matrix(all_locations)  # 计算距离矩阵

        # 为每个配送中心规划路径
        for center_idx, center in enumerate(centers):
            best_individual, best_fitness = genetic_algorithm(num_orders=len(ORDERS),
                                                              distance_matrix=distance_matrix,
                                                              max_carry=N, max_distance=MAX_DISTANCE,
                                                              pop_size=15000, generations=1000, mutation_rate=0.01)

            # 打印当前配送中心路径规划结果
            print(f"【配送中心{center}调度方案】")
            print("   *遗传算法初始化策略：")
            processed_orders = [order_locations[i - len(centers)] for i in best_individual if i >= len(centers)]
            # 去除processed_orders重复元素
            processed_orders = list(dict.fromkeys(processed_orders))
            # start_center_idx = center_idx
            drones = []

            # 计算每个无人机的路径和负责的订单
            for i in range(len(processed_orders)):
                DRONE_ID_COUNTER += 1
                tar_location = processed_orders[i]
                tar_location_idx = (all_locations).index(tar_location)

                # 计算路径距离
                distance = distance_matrix[center_idx][tar_location_idx] * 2
                # 统计ORDERS中所有location树形=tar_location的元素数量
                orders = []
                for order in ORDERS:
                    if order['location'] == tar_location:
                        orders.append(order)
                orders = orders[:N]
                if distance <= MAX_DISTANCE:
                    drone = Drone(DRONE_ID_COUNTER, orders, [tar_location], distance)
                    drones.append(drone)

            count_b4 = len(drones)
            drones = sorted(drones, key=lambda x: len(x.orders), reverse=True)
            print_drones(drones, center)
            # 注意：优化前后的路径都要记录（因为可能出现合并前较优而合并后较差）
            # for drone in drones:
            #     CHOICES.append(([order['id'] for order in drone.orders], drone.id, drone.distance.__round__(2)))
            add_choices(drones, center)
            # 贪心算法合并路径
            merge_drones(drones, all_locations, distance_matrix)
            # 按订单orders的数量排序
            drones = sorted(drones, key=lambda x: len(x.orders), reverse=True)
            print("   *贪心算法优化策略：")
            if len(drones) == count_b4:
                print("    无")
            else:
                print_drones(drones, center)
            add_choices(drones, center)
            # for drone in drones:
            #     CHOICES.append(([order['id'] for order in drone.orders], drone.id, drone.distance.__round__(2)))
        # for ech in CHOICES:
        #     print(ech)
        # 先根据CHOICES中据距离顺序排序，再根订单数倒序排序（等价于优先订单数再距离）
        CHOICES = sorted(CHOICES, key=lambda x: x[3])
        CHOICES = sorted(CHOICES, key=lambda x: len(x[1]), reverse=True)
        print(f"【系统调度最佳方案】")
        # print('CHOICES', CHOICES)
        while CHOICES:
            # 选择最优方案
            best_choice = CHOICES.pop(0)
            total_dis += best_choice[3]
            # 描述最佳方案
            print(
                f"    分配中心{best_choice[-1]}调度无人机[{best_choice[0]}]派送订单：{best_choice[1]}, 路径：{best_choice[2]}, 距离：{best_choice[3]} km")
            best_orders = best_choice[1]
            LOG.append(best_choice)  # 记录决策
            DELIVERED += best_orders  # 记录已配送订单
            # 移除最佳方案的互斥方案
            # 即移除CHOICES中订单对应的集合和DELIVERED中订单对应的集合有交集的元素
            for choice in CHOICES[::-1]:
                if set(choice[1]) & set(DELIVERED):
                    CHOICES.remove(choice)
        # 更新订单
        ORDERS = [order for order in ORDERS if order['id'] not in DELIVERED]
        # 紧急调度
        # 如果发现剩余订单有优先级低于20的，立即找到最近的配送中心加派无人机
        for order in ORDERS:
            if order['priority'] <= 20:
                min_dis = float('inf')
                min_center = None
                for center in centers:
                    dis = distance_matrix[all_locations.index(center)][all_locations.index(order['location'])]
                    if dis < min_dis:
                        min_dis = dis
                        min_center = center
                DRONE_ID_COUNTER += 1
                DELIVERED.append(order['id'])
                total_dis += min_dis * 2
                print(f"【紧急调度】")
                print(
                    f"    分配中心{min_center}调度无人机[{DRONE_ID_COUNTER}]派送订单：{order['id']}, 路径：[{min_center}, {order['location'], min_center}], 距离：{min_dis * 2} km")
                ORDERS.remove(order)
        print(f"   *当前周期共计：{total_dis:.2f} km")
        TOTAL_DISTANCE += total_dis
        print(f"   *累计（0~{current_time + T}分钟）共计：{TOTAL_DISTANCE:.2f} km")
        # 描述剩余订单
        print(f"【剩余订单】")
        for order in ORDERS:
            print(
                f"    订单[{order['id']}]：卸货点{order['location']}, 优先级{order['priority']}")

        print()


# 打印所有常量
def print_config():
    print('-' * bar_count)
    print(f"总时间（分钟）：{TOTAL_TIME}")
    print(f"周期（分钟）：{T}")
    print(f"配送中心数量：{J}")
    print(f"卸货点数量：{K}")
    print(f"最大新订单数（每卸货点*周期）：{MAX_ORDER_NUM}")
    print(f"配送空间（公里）：{W}*{W}")
    print(f"订单优先级（分钟或公里）：{P}")
    print(f"无人机最大载荷量：{N}")
    print(f"无人机最大续航（公里）：{MAX_DISTANCE}")


# 主函数
def main():
    print_config()
    route_planning()


if __name__ == "__main__":
    main()
