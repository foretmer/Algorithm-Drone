import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
from deap import base, creator, tools, algorithms

# 假设设置
centersNum = 1  # 配送中心数量
customerNum = 5  # 卸货点数量
ordersMax = 5  # 每个时间段最大订单数
timeInterval = 60  # 时间间隔（分钟）
deliveryMax = 20  # 无人机飞行距离限制（公里）
loadMax = 3  # 无人机最大负载（物品数）
speed = 60  # 无人机速度（公里/小时）

# 遗传算法设置
POP_SIZE = 50  # 种群大小
GEN = 30  # 迭代代数
CX_PB = 0.7  # 交叉概率
MUT_PB = 0.2  # 变异概率


# 生成随机位置
def generatePosition(count, size1, size2):
    return [(random.randint(size1, size2), random.randint(size1, size2)) for _ in range(count)]

# 位置距离计算
def distance(pos1, pos2):
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

# 生成的位置
centers = generatePosition(centersNum, 3, 7)
customers = generatePosition(customerNum, 0, 10)


# 订单类
class Order:
    def __init__(self, priority, customerId, generateTime):
        self.priority = priority  # 1: 一般, 2: 较紧急, 3: 紧急
        self.customerId = customerId
        self.generateTime = generateTime
        self.deadline = self.setDeadline(generateTime, priority)
        self.processed = False

    def setDeadline(self, generateTime, priority):
        if priority == 1:
            return generateTime + 180  # 一般，3小时内送达
        elif priority == 2:
            return generateTime + 90  # 较紧急，1.5小时内送达
        else:
            return generateTime + 30  # 紧急，0.5小时内送达

# 生成订单
def generateOrders(time):
    orderNum = random.randint(0, ordersMax)
    return [Order(random.randint(1, 3), random.randint(0, customerNum - 1), time) for _
            in range(orderNum)]

def createRoute(individual, center, customers):
    # 根据遗传算法个体生成路径
    route = [center] + [customers[i] for i in individual] + [center]
    return route

def evalRoute(individual, center, customers):
    # 计算路径的总距离
    route = createRoute(individual, center, customers)
    total_distance = sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1))
    return (total_distance,)

def setupGA(center, customerIds, customers):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # 确保随机抽样的元素数量不超过列表大小
    sample_size = min(len(customerIds), len(customerIds))
    toolbox.register("indices", random.sample, range(len(customerIds)), sample_size)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalRoute, center=center, customers=customers)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def optimizeRoutes(center, orders, customers):
    customerIds = [order.customerId for order in orders if not order.processed]
    if len(customerIds) < 2:  # 如果小于2个订单，则无法进行有效的路径规划
        return [center] + [customers[idx] for idx in customerIds] + [center]

    toolbox = setupGA(center, customerIds, customers)
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB, ngen=GEN, stats=stats, halloffame=hof, verbose=False)

    bestRoute = createRoute(hof[0], center, customers)
    for idx in hof[0]:
        orders[idx].processed = True
    return bestRoute


def selectDrones(orders, centers, customers):
    paths = []
    orders_by_center = defaultdict(list)

    for order in orders:
        if order.processed:
            continue
        customerPos = customers[order.customerId]
        nearest_center = min(centers, key=lambda x: distance(x, customerPos))
        orders_by_center[nearest_center].append(order)

    for center, center_orders in orders_by_center.items():
        if center_orders:
            best_path = optimizeRoutes(center, center_orders, customers)
            paths.append(best_path)
    return paths

# 绘图
def plot(centers, customers, paths):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(paths)))

    center_plotted, drop_plotted = False, False

    for x, y in centers:
        if not center_plotted:
            plt.scatter(x, y, c='blue', alpha=1, marker=',', linewidths=3)
            center_plotted = True
        else:
            plt.scatter(x, y, c='blue', alpha=1, marker=',', linewidths=3)

    for x, y in customers:
        if not drop_plotted:
            plt.scatter(x, y, c='red', alpha=1, marker='o', linewidths=3)
            drop_plotted = True
        else:
            plt.scatter(x, y, c='red', alpha=1, marker='o', linewidths=3)

    if paths:  # 确保路径列表不为空
        for idx, path in enumerate(paths):
            # 期望路径格式：路径是一个包含起点、途径点和终点的列表
            print(f"Path {idx + 1}: {path}")
            plt.plot([p[0] for p in path], [p[1] for p in path],
                     color=colors[idx], linestyle='--')

    plt.legend(loc='best')
    plt.title('Drone Delivery Routing')
    plt.grid(False)
    plt.show()

# 使用改进的路径选择逻辑
def simulation(duration):
    orders = []
    time = 0
    paths = []

    while time < duration:
        new_orders = generateOrders(time)
        orders.extend(new_orders)

        new_paths = selectDrones(orders, centers, customers)
        paths.extend(new_paths)

        plot(centers, customers, paths)
        plt.pause(1)

        time += timeInterval

    plt.show()


# 运行模拟
simulation(240)
