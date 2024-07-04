import os
import random
import matplotlib.pyplot as plt
from collections import deque
import pickle
import math
import shutil

# 确定的常量
MAX_DISTANCE = 10  # 卸货节点距配送中心的最大距离
DAY_TIME = 1440  # 一天的总分钟数
DRONE_MAX_FLIGHT = 20  # 无人机一次飞行最远路程
DRONE_SPEED = 1  # 无人机速度

# 自定义变量
J = 5  # 配送中心数
K = 20  # 卸货点数
T = 30  # 任务与下达命令的时间间隔
M = 5  # 卸货点每次生成的最大订单数
N = 10  # 每个无人机最多携带的物品数
MAP_LENGTH = 30  # 地图最大长度
MARGIN = 5  # 配送中心距离地图的边界
MIN_INTERVAL = 5  # 配送中心之间的最短间隔

# 遗传算法超参数
POPULATION = 50  # 种群大小
GENERATION = 1000  # 迭代次数
MUTATION_RATE = 0.1  # 变异概率
DELTA = 0.5  # 适应度多样性阈值


# 配送中心
class DistributionCenter:
    def __init__(self, id, coords=(0, 0)):
        self.id = id
        self.coords = coords


# 随机生成订单
def random_order():
    priority = random.randint(1, 6)
    if priority in (1, 2, 3):
        return 180
    elif priority in (4, 5):
        return 90
    elif priority == 6:
        return 30


# 卸货点
class UnloadingPoint:
    def __init__(self, id, coords=(0, 0)):
        self.id = id
        self.coords = coords
        self.orders = []
        self.nearest_distribution_center_id = None
        self.nearest_center_distance = None

    # 更新节点
    def update_point(self, m, t):
        for index, order in enumerate(self.orders):
            self.orders[index] -= t

        n = random.randint(0, m)
        for i in range(n):
            self.orders.append(random_order())
        self.orders.sort()

    # 获取优先订单
    def get_priority_orders(self):
        return [order for order in self.orders if order <= T]

    # 已完成的订单
    def finish_order(self, finished_orders):
        for finished_order in finished_orders:
            self.orders.remove(finished_order)


# 地图
class Map:
    def __init__(self):
        self.distribution_center_coords = []  # 配送中心坐标
        self.unloading_point_coords = []  # 卸货点坐标
        self.distribution_centers = []  # 配送中心实例
        self.unloading_points = []  # 卸货点实例
        self.distance_matrix = {"DC2UP": [], "UP2UP": []}  # 距离矩阵

    # 加载地图
    def load_map(self, map_path, j_num, k_num):
        # 若存在已生成的地图，读取之
        if os.path.exists(map_path):
            with open(map_path, "rb") as file:
                map = pickle.load(file)

            self.distribution_center_coords = map["distribution_center_coords"]
            self.unloading_point_coords = map["unloading_point_coords"]

        # 否则生成新地图
        else:
            self.generate_map(j_num, k_num, map_path)

        # 初始化配送中心和卸货点
        self.init_distribution_center()
        self.init_unloading_point()

        # 计算配送中心和卸货点之间的距离，初始化距离矩阵
        self.calculate_distance_matrix()

    # 生成地图
    def generate_map(self, j_num, k_num, map_path):
        dc_coords = []
        up_coords = []
        while len(dc_coords) < j_num:
            x, y = random.uniform(MARGIN, MAP_LENGTH - MARGIN), random.uniform(MARGIN, MAP_LENGTH - MARGIN)
            acceptable = True
            for center in dc_coords:
                # 保证配送中心之间的间隔
                if math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) < MIN_INTERVAL:
                    acceptable = False
                    break
            if acceptable:
                dc_coords.append((x, y))

        while len(up_coords) < k_num:
            x, y = random.uniform(0, MAP_LENGTH), random.uniform(0, MAP_LENGTH)
            for center in dc_coords:
                # 保证卸货点至少在一个装货点附近
                if math.sqrt(((x - center[0]) ** 2 + (y - center[1])) ** 2) < MAX_DISTANCE:
                    up_coords.append((x, y))
                    break

        self.distribution_center_coords = dc_coords
        self.unloading_point_coords = up_coords

        # 保存地图
        map = {"distribution_center_coords": self.distribution_center_coords,
               "unloading_point_coords": self.unloading_point_coords}

        with open(map_path, "wb") as file:
            pickle.dump(map, file)

        # 展示地图
        plt.figure(figsize=(10, 10))
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.scatter(*zip(*self.distribution_center_coords), s=100, color="blue", marker="*", label="配送中心")
        plt.scatter(*zip(*self.unloading_point_coords), s=50, color="red", marker="s", label="卸货点")
        for index, dc_coords in enumerate(self.distribution_center_coords):
            plt.annotate(str(index), [coord + 0.2 for coord in dc_coords])
        for index, up_coords in enumerate(self.unloading_point_coords):
            plt.annotate(str(index), [coord + 0.2 for coord in up_coords])
        plt.title("配送中心与卸货点地图")
        plt.legend()
        plt.grid(True)
        plt.savefig("./map.png")
        plt.show()

    # 初始化配送中心
    def init_distribution_center(self):
        for index, dc_coords in enumerate(self.distribution_center_coords):
            self.distribution_centers.append(DistributionCenter(index, dc_coords))

    # 初始化卸货点
    def init_unloading_point(self):
        for index, up_coords in enumerate(self.unloading_point_coords):
            self.unloading_points.append(UnloadingPoint(index, up_coords))

    # 计算距离矩阵
    def calculate_distance_matrix(self):
        # 计算配送中心到各个卸货点的距离
        for index, dc_coord in enumerate(self.distribution_center_coords):
            self.distance_matrix["DC2UP"].append([])
            for up_coord in self.unloading_point_coords:
                self.distance_matrix["DC2UP"][index].append(
                    math.sqrt((dc_coord[0] - up_coord[0]) ** 2 + (dc_coord[1] - up_coord[1]) ** 2))

        # 计算各个卸货点之间的距离
        for index, up_coord_1 in enumerate(self.unloading_point_coords):
            self.distance_matrix["UP2UP"].append([])
            for up_coord_2 in self.unloading_point_coords:
                self.distance_matrix["UP2UP"][index].append(
                    math.sqrt((up_coord_1[0] - up_coord_2[0]) ** 2 + (up_coord_1[1] - up_coord_2[1]) ** 2))

        # 计算距离每个卸货点最近的配送中心及其距离
        for up_index, up in enumerate(self.unloading_points):
            shortest_distance = math.inf
            nearest_id = 0
            for dc_index in range(J):
                if self.distance_matrix["DC2UP"][dc_index][up_index] < shortest_distance:
                    nearest_id = dc_index
                    shortest_distance = self.distance_matrix["DC2UP"][dc_index][up_index]

            up.nearest_distribution_center_id = nearest_id
            up.nearest_center_distance = shortest_distance


# 路径
class Routes:
    def __init__(self, routes):
        if routes is None:
            routes = []
        self.total_distance = 0
        self.sub_routes = routes

    # 打印路径
    def print_routes(self, unloading_destinations):
        for sub_route in self.sub_routes:
            center_id = unloading_destinations[sub_route[0]].nearest_distribution_center_id
            print("无人机飞行路径：" + "配送中心" + str(distribution_starts[center_id].id) + "->" + "".join(
                ["卸货点" + str(unloading_destinations[index].id) + "->" for index in sub_route]) + "配送中心" + str(
                distribution_starts[center_id].id))

    # 保存路径
    def save_routes(self, map, current_time):
        plt.figure(figsize=(10, 10))
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.scatter(*zip(*map.distribution_center_coords), s=100, color="blue", marker="*", label="配送中心")
        plt.scatter(*zip(*map.unloading_point_coords), s=50, color="red", marker="s", label="卸货点")
        for index, dc_coords in enumerate(map.distribution_center_coords):
            plt.annotate(str(index), [coord + 0.2 for coord in dc_coords])
        for index, up_coords in enumerate(map.unloading_point_coords):
            plt.annotate(str(index), [coord + 0.2 for coord in up_coords])
        plt.title(f"无人机飞行路径 @ {current_time} min")
        plt.legend()
        plt.grid(True)

        for sub_route in self.sub_routes:
            dc = map.distribution_centers[map.unloading_points[sub_route[0]].nearest_distribution_center_id]
            ups = [map.unloading_points[index] for index in sub_route]

            nodes = [dc]
            nodes.extend(ups)

            for index, node in enumerate(nodes):
                plt.plot([node.coords[0], nodes[(index + 1) % (len(nodes))].coords[0]],
                         [node.coords[1], nodes[(index + 1) % (len(nodes))].coords[1]])
        plt.savefig(f"./output/Drone Flight Route @ {current_time} min")


# 节约算法
def savings_algorithm(unloading_destinations, priority_list):
    priority_points = priority_list[0]
    print("优先级最高的卸货点：", priority_points)
    routes = [[up] for up in priority_points]
    savings = []

    for index, ud1 in enumerate(unloading_destinations):
        if index < len(unloading_destinations) - 1:
            for ud2 in unloading_destinations[index + 1:]:
                nearest_center_id = ud1.nearest_distribution_center_id
                saving = (map.distance_matrix["DC2UP"][nearest_center_id][ud1.id] +
                          map.distance_matrix["DC2UP"][nearest_center_id][ud2.id] -
                          map.distance_matrix["UP2UP"][ud1.id][ud2.id])
                savings.append((saving, ud1.id, ud2.id, nearest_center_id))

    savings.sort(reverse=True)

    # 约束1：无人机一次最多只能携带n个物品
    # 约束2：无人机一次飞行最远路程为20公里
    # 约束3：所有卸货点的需求都必须满足
    while len(savings) > 0:
        saving, ud1, ud2, nearest_center_id = savings[0]
        savings = savings[1:]
        route1 = None
        route2 = None

        for route in routes:
            if ud1 == route[-1]:
                route1 = route
            if ud2 == route[0]:
                route2 = route

        if route1 is None or route2 is None:
            continue

        if route1 is not None and route2 is not None and route1 != route2:
            new_route = route1 + route2
            total_items = sum(priority_list[1][point_id] for point_id in new_route)
            if total_items > N:
                continue

            center_id = unloading_destinations[new_route[0]].nearest_distribution_center_id
            distance_list = [map.distance_matrix["DC2UP"][center_id][new_route[0]]]
            for k in range(len(new_route) - 1):
                distance_list.append(map.distance_matrix["UP2UP"][new_route[k]][new_route[k + 1]])
            distance_list.append(map.distance_matrix["DC2UP"][center_id][new_route[-1]])
            total_distance = sum(distance_list)
            if total_distance > DRONE_MAX_FLIGHT:
                continue

            flight_time = [(sum(distance_list[:k + 1]) / DRONE_SPEED) for k in range(len(new_route))]
            constraint_list = [priority_list[2][node][0] for node in new_route]
            if sum([flight_time[i] > constraint_list[i] for i in range(len(constraint_list))]):
                continue

            routes.remove(route1)
            routes.remove(route2)
            routes.append(new_route)

            # 更新节约值
            for saving, ud1, ud2, center in savings:
                if ud1 in new_route and ud2 in new_route:
                    savings.remove((saving, ud1, ud2, center))
                elif ud2 in new_route[1:]:
                    savings.remove((saving, ud1, ud2, center))
                elif ud1 in new_route[:-1]:
                    savings.remove((saving, ud1, ud2, center))
                elif ud1 == new_route[-1]:
                    savings.remove((saving, ud1, ud2, center))
                    saving = (map.distance_matrix["DC2UP"][nearest_center_id][ud1] +
                              map.distance_matrix["DC2UP"][nearest_center_id][ud2] -
                              map.distance_matrix["UP2UP"][ud1][ud2])
                    savings.append((saving, ud1, ud2, nearest_center_id))
            savings.sort(reverse=True)

    return Routes(routes)


# 生成子代
def generate_child(child, parent1, parent2):
    N = len(parent1)
    child1, child2 = [0] * N, [0] * N
    flag1, flag2 = [0] * K, [0] * K

    i = random.randint(0, N - 1)
    j = i
    while j == i:
        j = random.randint(0, N - 1)
    if j < i:
        i, j = j, i

    for k in range(i, j + 1):
        child1[k] = parent1[k]
        flag1[child1[k]] = True
        child2[k] = parent2[k]
        flag2[child2[k]] = True

    index = (j + 1) % N
    p1, p2 = index, index
    while index != i:
        while flag1[parent2[p2]]:
            p2 = (p2 + 1) % N
        while flag2[parent1[p1]]:
            p1 = (p1 + 1) % N
        child1[index] = parent2[p2]
        flag1[child1[index]] = True
        child2[index] = parent1[p1]
        flag2[child2[index]] = True
        index = (index + 1) % N

    if random.random() < 0.5:
        child[:] = child1
    else:
        child[:] = child2


# 路径解码
def permutation_to_routes(permutation):
    global total_items, distance_list, flight_time
    routes = []
    current_route = None
    for up in permutation:
        unloading_destination = unloading_destinations[up]
        if not current_route:
            current_route = [up]
            center_id = unloading_destination.nearest_distribution_center_id
            total_items = len(unloading_destination.orders)
            distance_list = [map.distance_matrix["DC2UP"][center_id][up]]
            flight_time = current_time + (distance_list[0] / DRONE_SPEED)

        else:
            total_items += len(unloading_destination.orders)
            if total_items > N:
                routes.append(current_route)
                current_route = [up]
                total_items = flight_time = 0
                distance_list = []
                continue

            distance_list.append(map.distance_matrix["UP2UP"][current_route[-1]][up])
            total_distance = sum(distance_list)

            if total_distance > DRONE_MAX_FLIGHT:
                routes.append(current_route)
                current_route = [up]
                continue

            flight_time += (map.distance_matrix["UP2UP"][current_route[-1]][up] / DRONE_SPEED)
            if flight_time > unloading_destination.orders[0]:
                routes.append(current_route)
                current_route = [up]
                continue

            current_route.append(up)
    routes.append(current_route)

    return Routes(routes)


# 变异个体
def mutate_individual(child, child_routes):
    global next_up_index
    best_cost = child_routes.total_distance

    random_up_index = random.randint(0, len(child) - 1)
    random_up = child[random_up_index]

    random_up_route = []
    for sub_route in child_routes.sub_routes:
        if random_up in sub_route:
            random_up_route = sub_route
            break

    v1 = v2 = 0
    if random_up_index > 0:
        v1 = child[random_up_index - 1]
    if random_up_index < len(child) - 1:
        v2 = child[random_up_index + 1]

    if v1 and v2 and random_up != random_up_route[0]:
        s1 = map.distance_matrix["UP2UP"][random_up][v1] + map.distance_matrix["UP2UP"][random_up][v2] - \
             map.distance_matrix["UP2UP"][v1][v2]
    else:
        s1 = map.distance_matrix["DC2UP"][unloading_destinations[random_up].nearest_distribution_center_id][
                 random_up] + map.distance_matrix["UP2UP"][random_up][v2] - map.distance_matrix["DC2UP"][
                 unloading_destinations[v2].nearest_distribution_center_id][v2]

    route_index = better_route_index = 0
    for sub_route in child_routes.sub_routes:
        if sub_route != random_up_route and sum([len(unloading_destinations[up].orders) for up in sub_route]) + \
                len(unloading_destinations[random_up].orders) <= N:
            for i in range(len(sub_route)):
                v1 = sub_route[0] if i == 0 else sub_route[i - 1]
                v2 = sub_route[i]
                new_route = sub_route[:i] + [random_up] + sub_route[i:]

                if i == 0:
                    center_id = unloading_destinations[random_up].nearest_distribution_center_id
                    distance_list = [map.distance_matrix["DC2UP"][center_id][random_up]]
                else:
                    center_id = unloading_destinations[sub_route[0]].nearest_distribution_center_id
                    distance_list = [map.distance_matrix["DC2UP"][center_id][sub_route[0]]]
                for k in range(len(new_route) - 1):
                    distance_list.append(
                        map.distance_matrix["UP2UP"][new_route[k]][new_route[k + 1]])
                distance_list.append(map.distance_matrix["DC2UP"][center_id][new_route[-1]])

                total_distance_tmp = sum(distance_list)
                if total_distance_tmp > DRONE_MAX_FLIGHT:
                    continue

                flight_time = [current_time + (sum(distance_list[:k + 1]) / DRONE_SPEED) for k in
                               range(len(new_route) - 1)]
                constraint_list = [unloading_destinations[k].orders[0] for k in new_route]
                if flight_time > constraint_list:
                    continue

                if i == 0:
                    s2 = unloading_destinations[random_up].nearest_center_distance + \
                         map.distance_matrix["UP2UP"][random_up][v2] - unloading_destinations[
                             sub_route[0]].nearest_center_distance
                else:
                    s2 = map.distance_matrix["UP2UP"][v1][random_up] + \
                         map.distance_matrix["UP2UP"][v2][random_up] - \
                         map.distance_matrix["UP2UP"][v1][v2]
                temp_cost = (best_cost - s1 - s2)

                if temp_cost < best_cost:
                    best_cost = temp_cost
                    next_up_index = i
                    better_route_index = route_index
        route_index += 1

    if best_cost < child_routes.total_distance:
        random_up_route.remove(random_up)
        child_routes.sub_routes[better_route_index].insert(next_up_index, random_up)
        if [] in child_routes.sub_routes:
            child_routes.sub_routes.remove([])


# 查找最短路径
class Search:
    def __init__(self, map):
        self.map = map

    # 计算优先处理的订单
    def get_priority_order_list(self):
        priority_points = []
        priority_orders = [0] * K
        priority_orders_detail = [[]] * K

        for unloading_destination in self.map.unloading_points:
            priority_order = unloading_destination.get_priority_orders()
            if priority_order:
                priority_points.append(unloading_destination.id)
                priority_orders_detail[unloading_destination.id] = priority_order
                priority_orders[unloading_destination.id] = len(priority_order)
        if priority_points:
            return priority_points, priority_orders, priority_orders_detail
        else:
            return None

    # 计算路线适应度
    def calculate_fitness(self, routes, payload=0):
        total_distance = 0
        total_items = 0
        for sub_route in routes.sub_routes:
            center_id = unloading_destinations[sub_route[0]].nearest_distribution_center_id
            total_distance += map.distance_matrix["DC2UP"][center_id][sub_route[0]]
            for i in range(len(sub_route) - 1):
                total_items += len(unloading_destinations[i].orders)
                total_distance += map.distance_matrix["UP2UP"][sub_route[i]][sub_route[i + 1]]
            total_items += len(unloading_destinations[sub_route[-1]].orders)
            total_distance += map.distance_matrix["DC2UP"][center_id][sub_route[-1]]

        routes.total_distance = total_distance
        # 适应度设置为运送每一个货物的平均距离
        return total_distance / (payload + 0.01)

    # 初始化种群
    def init_population(self):
        population = []
        priority_list = self.get_priority_order_list()

        # 生成精英解
        elite_solution = savings_algorithm(unloading_destinations, priority_list)
        elite_permutation = []
        for sub_route in elite_solution.sub_routes:
            elite_permutation += sub_route
        fitness = self.calculate_fitness(elite_solution)
        population.append((elite_permutation, fitness))

        # 生成剩余解
        delta = DELTA
        count = 0
        while len(population) < POPULATION:
            individual = elite_permutation[:]
            random.shuffle(individual)
            routes = permutation_to_routes(individual)
            fitness = self.calculate_fitness(routes)

            # 检查多样性
            if all(abs(fitness - individual[1]) >= delta for individual in population):
                population.append((individual, fitness))
            if count >= GENERATION:
                count = 0
                delta -= 0.1
            count += 1
        return population


if __name__ == "__main__":
    # 确保输出目录存在
    output_dir = "output"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    map = Map()
    map.load_map("map.pkl", J, K)
    search = Search(map)

    distribution_starts = map.distribution_centers
    unloading_destinations = map.unloading_points

    current_orders = deque()
    current_time = 0
    total_distance = 0
    time_list = []
    total_distance_list = []
    while current_time <= DAY_TIME:
        time_list.append(current_time)
        print("\n" * 3)
        print("-" * 50)
        print(f"当前时间：{current_time}/{DAY_TIME}min")
        for point in unloading_destinations:
            # 更新订单情况
            point.update_point(M, T)
            # 打印订单情况
            print(f"卸货点{point.id}现有订单：{point.orders}")

        priority_points = search.get_priority_order_list()
        if priority_points is None:
            current_time += T
            total_distance_list.append(total_distance)
            continue

        # 初始化种群
        population = search.init_population()
        best_solution = population[0][0]
        best_fitness = float('inf')

        delta = DELTA

        # 主循环
        if len(best_solution) > 1:
            for generation in range(GENERATION):
                count = 0
                while True:
                    tournament = random.sample(population, max((len(population) // 2), 2))
                    tournament.sort(key=lambda x: x[1])
                    parent1, parent2 = tournament[:2]
                    parent1 = parent1[0]
                    parent2 = parent2[0]
                    child = [0] * len(parent1)
                    generate_child(child, parent1, parent2)
                    child_routes = permutation_to_routes(child)
                    fitness = search.calculate_fitness(child_routes)

                    # 局部搜索改进
                    if random.random() < MUTATION_RATE:
                        mutate_individual(child, child_routes)
                    # 检查多样性
                    if all(abs(fitness - ind[1]) >= delta for ind in population):
                        population.append((child, fitness))
                        break
                    else:
                        count += 1
                        if count >= GENERATION:
                            count = 0
                            delta -= 0.1

                # 选出被淘汰的个体
                population.sort(key=lambda ind: ind[1])
                population = population[:POPULATION]

            best_solution = population[0][0]
        best_routes = permutation_to_routes(best_solution)

        # 删去不必要的路径
        final_routes = []
        for route in best_routes.sub_routes:
            if route and any(point in priority_points[0] for point in route):
                nearest_center_id = unloading_destinations[route[0]].nearest_distribution_center_id
                final_routes.append(route)

        best_routes4 = Routes(final_routes)
        best_routes.save_routes(map, current_time)
        search.calculate_fitness(best_routes)
        best_routes.print_routes(unloading_destinations)
        total_distance += best_routes.total_distance
        total_distance_list.append(total_distance)

        priority_order_list = search.get_priority_order_list()
        for up in best_solution:
            unloading_destinations[up].finish_order(priority_order_list[2][up])

        current_time += T

    print("所有无人机总飞行路程：", total_distance)
