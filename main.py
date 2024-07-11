import numpy as np
import random
from deap import creator, base, tools, algorithms
from datetime import timedelta, datetime

# 定义全局变量
CAPACITY = 20  # 最大飞行距离
MAX_LOAD = 3  # 最大携带数量
delivery_centers = [(0, 0), (5, 9), (10, 0)]
unload_points = [(1, 4), (7, 5), (9, 3), (7, 8), (5, 2),(3, 9),(1, 8),(5, 6),(0, 2),(10, 5),(1, 10)]
order_priorities = {'critical': 0.5, 'urgent': 1.5, 'normal': 3}  # 单位：小时

# 生成随机订单，包括时间戳和优先级
def generate_orders(unload_points, start_time):
    orders = []
    for point in unload_points:
        num_orders = np.random.randint(0, 3)  # 每个点生成0到3个订单
        for _ in range(num_orders):
            priority = np.random.choice(['normal', 'urgent', 'critical'], p=[0.5, 0.3, 0.2])
            order_time = start_time + timedelta(hours=order_priorities[priority])
            orders.append((point, order_time))
    return orders

def calc_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def optimize_route(center, orders):
    if not orders:
        return []

    optimized_orders = []
    current_location = center
    remaining_orders = orders[:]

    while remaining_orders:
        # 找到离当前位置最近的订单
        next_order = min(remaining_orders, key=lambda x: calc_distance(current_location, x[0]))
        optimized_orders.append(next_order)
        current_location = next_order[0]
        remaining_orders.remove(next_order)

    return optimized_orders

def evaluate(individual):
    total_distance = 0
    for drone in individual:
        center = drone['center']
        drone['orders'] = optimize_route(center, drone['orders'])
        route = drone['orders']
        if len(route) > 0:
            route_distance = calc_distance(center, route[0][0])
            for i in range(len(route) - 1):
                route_distance += calc_distance(route[i][0], route[i + 1][0])
            route_distance += calc_distance(route[-1][0], center)
            if route_distance > CAPACITY or len(route) > MAX_LOAD:
                return 10000, 
        else:
            route_distance = 0
        total_distance += route_distance
    return total_distance,

# 创建适应度函数和个体类
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual(orders):
    individual = []
    available_orders = orders[:]  # 创建订单的副本，避免修改原始列表
    while len(available_orders) > 0:
        center = random.choice(delivery_centers)
        assigned_orders = random.sample(available_orders, min(np.random.randint(0, MAX_LOAD), len(available_orders)))
        individual.append({'center': center, 'orders': assigned_orders})
        available_orders = [order for order in available_orders if order not in assigned_orders]
    return creator.Individual(individual)  # 包装为 creator.Individual 类型

# 确保订单数量不变
def fix_order_counts(ind, original_orders):
    # 获取每个订单的初始计数
    order_count = {order: original_orders.count(order) for order in original_orders}
    # 计算当前订单的数量
    current_order_count = {}
    for drone in ind:
        for order in drone['orders']:
            if order in current_order_count:
                current_order_count[order] += 1
            else:
                current_order_count[order] = 1

    # 修复订单数量
    for order, count in order_count.items():
        if order in current_order_count:
            current_count = current_order_count[order]
        else:
            current_count = 0

        if current_count < count:
            for _ in range(count - current_count):
                for drone in ind:
                    if len(drone['orders']) < MAX_LOAD:
                        drone['orders'].append(order)
                        break
        elif current_count > count:
            for _ in range(current_count - count):
                for drone in ind:
                    if order in drone['orders']:
                        drone['orders'].remove(order)
                        break
    return ind

def crossover(ind1, ind2):
    # 执行部分匹配交叉（PMX）
    temp_ind1 = [order for drone in ind1 for order in drone['orders']]
    temp_ind2 = [order for drone in ind2 for order in drone['orders']]
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size - 1)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint1 > cxpoint2:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    temp1 = [drone['orders'] for drone in ind1[cxpoint1:cxpoint2]]
    temp2 = [drone['orders'] for drone in ind2[cxpoint1:cxpoint2]]

    for i in range(cxpoint1, cxpoint2):
        ind1[i]['orders'], ind2[i]['orders'] = temp2[i - cxpoint1], temp1[i - cxpoint1]

    ind1 = fix_order_counts(ind1, temp_ind1)
    ind2 = fix_order_counts(ind2, temp_ind2)

    return ind1, ind2

def mutate(individual):
    mutation_type = random.choice(['move', 'swap', 'center'])  # 加入中心变异作为一个选项

    if mutation_type == 'move':
        # 随机选择一个无人机，并从其订单列表中移除一个订单
        from_drone = random.choice(individual)
        if from_drone['orders']:
            order = random.choice(from_drone['orders'])
            from_drone['orders'].remove(order)
            to_drone = random.choice(individual)
            to_drone['orders'].append(order)

    elif mutation_type == 'swap':
        # 随机选择两个无人机进行订单交换
        if len(individual) > 1:
            from_drone, to_drone = random.sample(individual, 2)  # 确保选择的是两个不同的无人机
            if from_drone['orders'] and to_drone['orders']:
                from_order = random.choice(from_drone['orders'])
                to_order = random.choice(to_drone['orders'])

                # 交换订单
                from_drone['orders'].remove(from_order)
                to_drone['orders'].remove(to_order)
                from_drone['orders'].append(to_order)
                to_drone['orders'].append(from_order)

    elif mutation_type == 'center':
        # 随机选择一个无人机并改变其起始中心
        drone = random.choice(individual)
        new_center = random.choice(delivery_centers)
        while new_center == drone['center']:  # 确保新中心与原中心不同
            new_center = random.choice(delivery_centers)
        drone['center'] = new_center

    return individual,

def calculate_total_flight_distance(flights):
    total_distance = 0
    for drone in flights:
        if drone['orders']:  # 确保有订单
            path = drone['orders']
            distance = calc_distance(drone['center'], path[0][0])
            for i in range(len(path) - 1):
                distance += calc_distance(path[i][0], path[i+1][0])
            distance += calc_distance(path[-1][0], drone['center'])  # 返回到起点
            total_distance += distance
    return total_distance

import matplotlib.pyplot as plt

def plot_flights(fig,flights,c_t, c_o, t_o,t):
    fig.clf()
    plt.ion()
    ax = fig.add_subplot(111)
    for drone in flights:
        if drone['orders']:
            path = [drone['center']] + [order[0] for order in drone['orders']] + [drone['center']]
            xs, ys = zip(*path)  # 解包每个订单点的x和y坐标
            ax.plot(xs, ys, marker='o')  # 绘制路径和点
            ax.text(drone['center'][0], drone['center'][1], 'Center', fontsize=12, ha='right')

    fig.suptitle(f'Flight Paths at {c_t}\n Orders({t_o - c_o} / {t_o})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([-1,11])
    ax.set_ylim([-1,11])
    ax.grid(True)
    plt.savefig(f'{t}.png')
    plt.pause(0.01)
    


def main():
    start_time = datetime.now()-datetime.now() # 定义开始时间
    end_time = start_time + timedelta(hours=24)  # 运行一天
    current_time = start_time
    orders = []
    fig, ax = plt.subplots()
    from_continue = False
    dis_list=[]
    order_num=[]
    
    while current_time < end_time:
        if not from_continue:
            orders.extend(generate_orders(unload_points, current_time))
            total_order = len(orders)
            popu = 50
            ngen = 100
            cxpb = 0.4
            mutpb = 0.3
        from_continue = False
        # 处理当前订单
        toolbox = base.Toolbox()
        toolbox.register("individual", create_individual, orders=orders)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        population = toolbox.population(n=popu)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)

        # 强制重新评估所有个体的适应度
        for ind in population:
            del ind.fitness.values

        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof,verbose=False)

        # 从HallOfFame中获取最优个体
        best_individual = hof[0]
        if best_individual.fitness.values[0] == 10000:
            print('-----------------------------')
            print(f'Best fitness: {best_individual.fitness.values[0]}')
            print(f'Remaining orders: {len(orders)}')
            print('Retry!')
            print('-----------------------------')
            from_continue = True
            popu += 10
            ngen += 10
            cxpb += 0.01
            mutpb += 0.01
            continue

        # 提取并删除在current_time + 30分钟内的订单
        next_time = current_time + timedelta(minutes=30)
        flights = []

        for drone in best_individual:
            for order in drone['orders']:
                if order[1] <= next_time:
                    flights.append(drone)
                    for sorder in flights[-1]['orders']:
                        orders.remove(sorder)
                    break

        # 如果剩余订单超过15，则额外安排平均配送距离短的航班
        if len(orders) > 15:
            # 计算每个无人机的平均飞行距离
            drone_distances = []
            for drone in best_individual:
                if drone not in flights and drone['orders']:  # 未安排且有订单的无人机
                    total_drone_distance = 0
                    num_orders = len(drone['orders'])
                    if num_orders > 0:
                        for i in range(num_orders - 1):
                            total_drone_distance += calc_distance(drone['orders'][i][0], drone['orders'][i + 1][0])
                        total_drone_distance += calc_distance(drone['center'], drone['orders'][0][0])
                        total_drone_distance += calc_distance(drone['orders'][-1][0], drone['center'])
                        average_distance = total_drone_distance / num_orders
                        drone_distances.append((drone, average_distance))

            # 选择平均距离短的航班
            drone_distances.sort(key=lambda x: x[1])  # 按平均距离排序
            for drone, _ in drone_distances:
                flights.append(drone)
                for order in drone['orders']:
                    if order in orders:
                        orders.remove(order)
                if len(orders) <= 15:
                    break
        dis_list.append(calculate_total_flight_distance(flights))
        order_num.append(total_order-len(orders))
        print('-----------------------------')
        print(f'Flights scheduled by {current_time}: {flights}')
        print(f'Total flights: {len(flights)}')
        print(f'Total flying distance: {calculate_total_flight_distance(flights)}')
        print(f'Remaining orders: {len(orders)}')
        current_time += timedelta(minutes=30)  # 更新时间
        print('-----------------------------')
        plot_flights(fig,flights,current_time,len(orders),total_order,current_time.total_seconds())
        # input()
    plt.figure()
    plt.plot(np.array(dis_list) / np.array(order_num))
    plt.xlim((-2,50))
    plt.ylim((0,20))
    plt.show()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()