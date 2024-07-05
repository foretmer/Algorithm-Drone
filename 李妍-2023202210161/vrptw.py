import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from gurobipy import GRB, Model, quicksum, tupledict, tuplelist
import numpy as np


def read_input(filename):
    """
    :param filename: 数据文件路径
    :return:
    """
    N = []  # 所有节点
    Q = {}  # 节点需求
    TT = {}  # 节点旅行时间
    ET = {}  # 节点最早开始服务时间
    LT = {}  # 节点最晚结束服务时间
    ST = {}  # 节点服务时间
    x_coord = {}  # 节点横坐标
    y_coord = {}  # 节点纵坐标
    Cost = {}
    df = pd.read_csv(filename)
    for i in range(df.shape[0]):
        id = df['id'][i]
        N.append(id)
        x_coord[id] = df['x_coord'][i]
        y_coord[id] = df['y_coord'][i]
        Q[id] = df['demand'][i]
        ET[id] = df['start_time'][i]
        LT[id] = df['end_time'][i]
        ST[id] = df['service_time'][i]
    for f_n in N:
        for t_n in N:
            dist = math.sqrt((x_coord[f_n] - x_coord[t_n]) ** 2 + (y_coord[f_n] - y_coord[t_n]) ** 2)
            Cost[f_n, t_n] = dist
            TT[f_n, t_n] = dist / 1  # 假设速度为1
    return N, Q, TT, ET, LT, ST, Cost, x_coord, y_coord


def build_model(N, Q, TT, ET, LT, ST, Cost, CAP, K, max_distance):
    """
    :param N: 所有节点集合，其中N[0]为车场
    :param Q: 节点需求集合
    :param TT: 旅行时间
    :param ET: 节点最早开始服务时间
    :param LT：节点最晚结束服务时间
    :param ST: 节点服务时间
    :param CAP: 车辆容量
    :param Cost: 旅行费用
    :param K: 车队
    :return:
    """
    C = N[1:]  # 需求节点
    M = 10 ** 5
    depot = N[0]
    # 创建模型
    vrptw_model = Model()
    # 添加变量
    X = vrptw_model.addVars(N, N, K, vtype=GRB.BINARY, name='X(i,j,k)')
    T = vrptw_model.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0, name='T[i,k]')
    # 设置目标函数
    z1 = quicksum(Cost[i, j] * X[i, j, k] for i in N for j in N for k in K if i != j)
    vrptw_model.setObjective(z1, GRB.MINIMIZE)
    # 车辆起点约束
    vrptw_model.addConstrs(quicksum(X[depot, j, k] for j in N) == 1 for k in K)
    # 车辆路径连续约束
    vrptw_model.addConstrs(
        quicksum(X[i, j, k] for j in N if j != i) == quicksum(X[j, i, k] for j in N if j != i) for i in C for k in K)
    # 车辆终点约束
    vrptw_model.addConstrs(quicksum(X[j, depot, k] for j in N) == 1 for k in K)
    # 需求服务约束
    vrptw_model.addConstrs(quicksum(X[i, j, k] for k in K for j in N if j != i) == 1 for i in C)
    # 车辆容量约束
    vrptw_model.addConstrs(quicksum(Q[i] * X[i, j, k] for i in C for j in N if i != j) <= CAP for k in K)
    # 时间窗约束
    vrptw_model.addConstrs(
        T[i, k] + ST[i] + TT[i, j] - (1 - X[i, j, k]) * M <= T[j, k] for i in C for j in C for k in K if i != j)
    vrptw_model.addConstrs(T[i, k] >= ET[i] for i in N for k in K)
    vrptw_model.addConstrs(T[i, k] <= LT[i] for i in N for k in K)
    # 续航里程约束
    vrptw_model.addConstrs(
        quicksum(Cost[i, j] * X[i, j, k] for i in N for j in N if i != j) <= max_distance for k in K)
    return vrptw_model, X, T


def draw_routes(route_list, x_coord, y_coord):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    center = (x_coord[0], y_coord[0])
    circle = plt.Circle(center, 10, color='blue', fill=False)
    ax.add_artist(circle)
    for route in route_list:
        path_x = []
        path_y = []
        for n in route[1:]:
            path_x.append(x_coord[n])
            path_y.append(y_coord[n])
        plt.plot(path_x, path_y, linewidth=0.5, marker='s', ms=5)
    plt.show()
    return True


def save_file(route_list, route_timetable, total_cost):
    wb = xlsxwriter.Workbook('路径方案.xlsx')
    ws = wb.add_worksheet()
    ws.write(0, 0, '总费用')
    ws.write(0, 1, total_cost)
    ws.write(1, 0, '车辆')
    ws.write(1, 1, '路径')
    ws.write(1, 2, '时刻')
    row = 2
    for id, route in enumerate(route_list):
        ws.write(row, 0, route[0])
        route_str = [str(i) for i in route[1:]]
        ws.write(row, 1, '-'.join(route_str))
        timetable_str = [str(i) for i in route_timetable[id]]
        ws.write(row, 2, '-'.join(timetable_str))
        row += 1
    wb.close()

def generate_distribution_centers(map_size, num_centers, radius):
    centers = []
    attempts = 0
    while len(centers) < num_centers and attempts < 1000:
        attempts += 1
        new_center = (round(random.uniform(radius, map_size - radius)), round(random.uniform(radius, map_size - radius)))
        if all(math.sqrt((new_center[0] - c[0]) ** 2 + (new_center[1] - c[1]) ** 2) >= 1.6 * radius for c in centers):
            centers.append(new_center)
    return centers


def generate_delivery_points(map_size, num_points, centers, radius):
    points = set()  # Use a set to avoid duplicates
    while len(points) < num_points:
        center = centers[len(points) % len(centers)]
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, radius)
        new_point = (round(center[0] + r * math.cos(angle)), round(center[1] + r * math.sin(angle)))
        if 0 <= new_point[0] <= map_size and 0 <= new_point[1] <= map_size and new_point not in centers:
            points.add(new_point)  # Add point to the set
    return list(points)


def plot_map(centers, points, radius, map_size):
    fig, ax = plt.subplots()
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect('equal', adjustable='box')

    for center in centers:
        circle = plt.Circle(center, radius, color='blue', fill=False)
        ax.add_artist(circle)
        ax.plot(center[0], center[1], 'bo')

    for point in points:
        ax.plot(point[0], point[1], 'ro')

    plt.show()

def get_route(centers, points, max_distance):
    N = []  # 所有节点
    Q = {}  # 节点需求
    TT = {}  # 节点旅行时间
    ET = {}  # 节点最早开始服务时间
    LT = {}  # 节点最晚结束服务时间
    ST = {}  # 节点服务时间
    x_coord = {}  # 节点横坐标
    y_coord = {}  # 节点纵坐标
    id = 0
    N.append(id)
    x_coord[id] = centers[0]
    y_coord[id] = centers[1]
    Q[id] = 0
    ET[id] = 0
    LT[id] = 9999
    ST[id] = 0
    for i, nodes in enumerate(points):
        id = i + 1
        N.append(id)
        x_coord[id] = points[i][0][0]
        y_coord[id] = points[i][0][1]
        Q[id] = points[i][1] # 节点需求量
        ET[id] = 0  # 开始服务时间
        LT[id] = points[i][2]  # 最晚服务时间
        ST[id] = 0
    """
    :param N: 所有节点集合，其中N[0]为车场
    :param Q: 节点需求集合
    :param TT: 旅行时间
    :param ET: 节点最早开始服务时间
    :param LT：节点最晚结束服务时间
    :param ST: 节点服务时间
    :param CAP: 车辆容量
    :param Cost: 旅行费用
    :param K: 车队
    """
    depot = N[0]
    CAP = 30
    Cost = {}
    K = list(range(1, NUM_POINTS))
    for f_n in N:
        for t_n in N:
            dist = math.sqrt((x_coord[f_n] - x_coord[t_n]) ** 2 + (y_coord[f_n] - y_coord[t_n]) ** 2)
            Cost[f_n, t_n] = dist
            TT[f_n, t_n] = dist / 1  # 假设速度为60
    vrptw_model, X, T = build_model(N, Q, TT, ET, LT, ST, Cost, CAP, K, max_distance)
    vrptw_model.Params.TimeLimit = 1500
    vrptw_model.setParam('OptimalityTol', 1e-2)
    vrptw_model.optimize()
    route_list = []
    route_timetable = []
    # 提取车辆路径
    for k in K:
        route = [depot]
        cur_node = depot
        cur_k = None
        for j in N[1:]:
            if X[depot, j, k].x > 0:
                cur_node = j
                cur_k = k
                route.append(j)
                N.remove(j)
                break
        if cur_k is None:
            continue
        while cur_node != depot:
            for j in N:
                if X[cur_node, j, cur_k].x > 0:
                    cur_node = j
                    route.append(j)
                    if j != depot:
                        N.remove(j)
                    break
        route.insert(0, k)
        route_list.append(route)

    for route in route_list:
        route = route[1:]
        total_cost = 0
        for point in range(len(route)-1):
            total_cost = total_cost + TT[(route[point],route[point+1])]
        print("route cost:" + str(total_cost))

    return route_list, vrptw_model.objVal

def cal_possi(x):
    x_min = 20
    x_max = 180
    y_min = 0.1
    y_max = 1
    k = -0.1  # 调整这个参数以改变函数的斜率
    x_mid = (x_min + x_max) / 2  # 中心点
    return y_min + (y_max - y_min) / (1 + np.exp(k * (-x + x_mid)))

if __name__ == '__main__':
    # Constants
    MAP_SIZE = 30
    NUM_CENTERS = 1
    NUM_POINTS = 7
    RADIUS = 10
    max_distance = 20
    max_demand = 5

    # Generate distribution centers and delivery points
    centers = generate_distribution_centers(MAP_SIZE, NUM_CENTERS, RADIUS)
    points = generate_delivery_points(MAP_SIZE, NUM_POINTS, centers, RADIUS)

    #plot_map(centers, points, RADIUS, MAP_SIZE)

    incident = []
    for center in centers:
        incident_points = []
        for i, point in enumerate(points):
            if math.sqrt((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2) <= 10:
                incident_points.append(i)
        incident.append(incident_points)

    # 聚类
    center_sub_classes = []
    for i, center in enumerate(centers):
        sub_class = []
        selected_points = [[points[ex], 1, 999] for ex in incident[i]]
        route_list, _ = get_route(center, selected_points, max_distance)
        for route in route_list:
            sub_class.append([incident[i][ex-1] for ex in route[2:-1]])
        center_sub_classes.append(sub_class)

    time = 0
    total_demand = []
    time_step = 5
    total_demand_new = [[ex[0], ex[1], ex[2]-time_step, ex[3]] for ex in total_demand]
    total_demand = total_demand_new
    for i in range(NUM_POINTS):
        random_demand = random.randint(1, max_demand)
        '''if (random_demand > 0) and (random.random() > 0.5):
            total_demand.append([points[i], random_demand, 30, i])
        random_demand = random.randint(0, max_demand)
        if (random_demand > 0) and (random.random() > 0.5):
            total_demand.append([points[i], random_demand, 90, i])
        random_demand = random.randint(0, max_demand)
        if (random_demand > 0) and (random.random() > 0.5):
            total_demand.append([points[i], random_demand, 180, i])'''
        if random_demand > 0:
            total_demand.append([points[i], random_demand, 30, i])
        random_demand = random.randint(0, max_demand)
        if random_demand > 0:
            total_demand.append([points[i], random_demand, 90, i])
        random_demand = random.randint(0, max_demand)
        if random_demand > 0:
            total_demand.append([points[i], random_demand, 180, i])
    total_demand_new = []
    for i in range(NUM_POINTS):
        time_limit = 180
        need = 0
        for demand in total_demand:
            if demand[3] == i:
                need = need + demand[1]
                if demand[2] < time_limit:
                    time_limit = demand[2]
        if need > 0:
            total_demand_new.append([points[i], need, time_limit, i])

    total_demand = total_demand_new
    possi_add = [0] * NUM_POINTS
    added = []
    demanded = [0] * NUM_POINTS
    demanded_points = [ex[0] for ex in total_demand]
    for i, point in enumerate(points):
        if point in demanded_points:
            demanded[i] = 1
    center_added_point = []
    for center_sub_class in center_sub_classes:
        added_point = []
        for sub_class in center_sub_class:
            total_sub_demanded = 0
            for point in sub_class:
                if demanded[point] == 1 and point not in added:
                    total_sub_demanded = total_sub_demanded + 1
            if total_sub_demanded > 2 or (total_sub_demanded / len(sub_class)) > 0.8:
                for point in sub_class:
                    if point not in added:
                        added_point.append(point)
                        added.append(point)
            elif total_sub_demanded > 1 and random.random() < 0.66:
                for point in sub_class:
                    if point not in added:
                        added_point.append(point)
                        added.append(point)
        center_added_point.append(added_point)

    for demand in total_demand:
        if random.random() < cal_possi(demand[2]):
            for i, incident_1 in enumerate(incident):
                if demand[3] in incident_1 and demand[3] not in added:
                    center_added_point[i].append(demand[3])

    opt = 0
    fig, ax = plt.subplots()
    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_aspect('equal', adjustable='box')
    for i, point in enumerate(points):
        if i in added:
            ax.plot(point[0], point[1], 'gx')
        else:
            ax.plot(point[0], point[1], 'ro')


    demand_after_deleted = []
    for i, center in enumerate(centers):
        added_point = center_added_point[i]
        selected_demands = []
        for demand in total_demand:
            if demand[3] in added_point:
                selected_demands.append(demand)
        total_demand = [demand for demand in total_demand if demand not in selected_demands]
        route_list, opt_center = get_route(center, selected_demands, max_distance)
        opt = opt + opt_center
        circle = plt.Circle(center, 10, color='blue', fill=False)
        ax.add_artist(circle)
        for route in route_list:
            path_x = []
            path_y = []
            path_x.append(center[0])
            path_y.append(center[1])
            for n in route[2:-1]:
                path_x.append(points[selected_demands[n-1][3]][0])
                path_y.append(points[selected_demands[n-1][3]][1])
            path_x.append(center[0])
            path_y.append(center[1])
            plt.plot(path_x, path_y, linewidth=0.5, marker='s', ms=5)


    plt.show()
    print("end")
    print("最优路径距离:", opt)



