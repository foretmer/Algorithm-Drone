# -*- coding: utf-8 -*-
import math
import random
import pandas as pd


import random

def generate_coordinates(num_centers, num_unloading_points):
    centers = []
    unloading_points = []
    # 生成配送中心坐标
    for _ in range(num_centers):
        center_x = random.randint(0, 50)  # 假设配送区域是一个50x50的区域
        center_y = random.randint(0, 50)
        centers.append((center_x, center_y))
    # 生成卸货点坐标
    for _ in range(num_unloading_points):
        while True:
            unloading_x = random.randint(0, 50)
            unloading_y = random.randint(0, 50)
            closest_center = min(centers, key=lambda c: distance(c, (unloading_x, unloading_y)))
            if distance(closest_center, (unloading_x, unloading_y)) <= 10:
                unloading_points.append((unloading_x, unloading_y))
                break
    return centers + unloading_points

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def generate_order(coordinates, DC,m):#DC为配送中心的数量，m为订单生成的最大数量
    demand = [0] * DC
    time = [0] * DC
    customers = coordinates[:DC]
    for i in range (DC-1,len(coordinates)):
        first = random.randint(0, m)
        if first > 0:
            customers.append(coordinates[i])
            demand.append(m)
            time.append(30)
        if m-first > 0:
            second = random.randint(0, m-first)
            if second > 0:
                customers.append(coordinates[i])
                demand.append(m)
                time.append(90)
            if m-first-second > 0:
                third = random.randint(0, m-first-second)
                if third > 0:
                    customers.append(coordinates[i])
                    demand.append(m)
                    time.append(180)
    return customers,time,demand



def calDistance(PointCoordinates):
    # 计算距离矩阵 输入坐标；输出距离矩阵dis_matrix
    dis_matrix = pd.DataFrame(data=None, columns=range(len(PointCoordinates)), index=range(len(PointCoordinates)))
    for i in range(len(PointCoordinates)):
        xi, yi = PointCoordinates[i][0], PointCoordinates[i][1]
        for j in range(len(PointCoordinates)):
            xj, yj = PointCoordinates[j][0], PointCoordinates[j][1]
            dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
    return dis_matrix


def assign_distribution_center(dis_matrix, DC, C):
    d = [[] for i in range(DC)]  # 存储分配的列表
    for i in range(DC, DC + C):
        d_i = [dis_matrix.loc[i, j] for j in range(DC)]  # 取出当前卸货点分别距离配送中心的距离
        min_dis_index = d_i.index(min(d_i))  # 取出最近的配送中心
        d[min_dis_index].append(i)  # 将卸货点点分配给配送中心
    return d


def greedy(PointCoordinates, dis_matrix, certer_number):
    dis_matrix = dis_matrix.iloc[[certer_number] + PointCoordinates, [certer_number] + PointCoordinates].astype(
        'float64')  # 只取当前需要的配送中心和卸货点
    for i in PointCoordinates: dis_matrix.loc[i, i] = math.pow(10, 10)
    dis_matrix.loc[:, certer_number] = math.pow(10, 10)  # 确保配送中心不在编码内
    line = []  # 初始化
    now_cus = random.sample(PointCoordinates, 1)[0]  # 随机生成出发点
    line.append(now_cus)  # 添加当前卸货点到路径
    dis_matrix.loc[:, now_cus] = math.pow(10, 10)  # 更新距离矩阵，已经过卸货点不再被取出
    for i in range(1, len(PointCoordinates)):
        next_cus = dis_matrix.loc[now_cus, :].idxmin()  # 距离最近的卸货点
        line.append(next_cus)  # 添加进路径
        dis_matrix.loc[:, next_cus] = math.pow(10, 10)  # 更新距离矩阵
        now_cus = next_cus  # 更新当前卸货点
    return line


def calFitness(birdPop, certer_number, Demand, dis_matrix, CAPACITY, DISTABCE, C0, C1, C2, time, V):
    birdPop_car, fits = [], []  # 初始化
    for j in range(len(birdPop)):
        bird = birdPop[j]
        lines = []  # 存储线路
        line = [certer_number]  # 每辆无人机配送卸货点，起点是配送中心
        dis_sum = 0  # 线路距离
        dis, d = 0, 0  # 当前卸货点距离前一个卸货点的距离、当前卸货点需求量
        i = 0  # 指向配送中心
        time_point = 0 #
        late = 0 # 迟到时间
        while i < len(bird):
            if line == [certer_number]:  # 无人机未分配客户点
                dis += dis_matrix.loc[certer_number, bird[i]]  # 记录距离
                line.append(bird[i])  # 为卸货点分配无人机
                d += Demand[bird[i]]  # 记录需求量
                time_point += dis_matrix.loc[0, bird[i]] / V
                if time_point > time[bird[i]]:
                    late = time_point - time[bird[i]]
                i += 1  # 指向下一个卸货点
            else:  # 已分配卸货点则需判断无人机载重和飞行距离
                if (dis_matrix.loc[line[-1], bird[i]] + dis_matrix.loc[bird[i], certer_number] + dis <= DISTABCE) & (
                        d + Demand[bird[i]] <= CAPACITY):
                    dis += dis_matrix.loc[line[-1], bird[i]]
                    time_point += dis_matrix.loc[line[-1], bird[i]] / V
                    if time_point > time[bird[i]]:
                        late = time_point - time[bird[i]]
                    line.append(bird[i])
                    d += Demand[bird[i]]
                    i += 1
                else:
                    dis += dis_matrix.loc[line[-1], certer_number]  # 当前无人机装满
                    line.append(certer_number)
                    dis_sum += dis
                    lines.append(line)
                    # 下一辆无人机
                    dis, d = 0, 0
                    line = [certer_number]
                    time_point = 0

        # 最后一辆无人机
        dis += dis_matrix.loc[line[-1], certer_number]
        line.append(certer_number)
        dis_sum += dis
        lines.append(line)

        birdPop_car.append(lines)
        fits.append(round(C1 * dis_sum + C0 * len(lines) + C2 * late, 1))

    return birdPop_car, fits, dis_sum


def crossover(bird, pLine, gLine, w, c1, c2):

    croBird = [None] * len(bird)  # 初始化
    parent1 = bird  # 选择parent1
    # 选择parent2（轮盘赌操作）
    randNum = random.uniform(0, sum([w, c1, c2]))
    if randNum <= w:
        parent2 = [bird[i] for i in range(len(bird) - 1, -1, -1)]  # bird的逆序
    elif randNum <= w + c1:
        parent2 = pLine
    else:
        parent2 = gLine

    # parent1-> croBird
    start_pos = random.randint(0, len(parent1) - 1)
    end_pos = random.randint(0, len(parent1) - 1)
    if start_pos > end_pos: start_pos, end_pos = end_pos, start_pos
    croBird[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()

    # parent2 -> croBird
    list2 = list(range(0, start_pos))
    list1 = list(range(end_pos + 1, len(parent2)))
    list_index = list1 + list2  # croBird从后往前填充
    j = -1
    for i in list_index:
        for j in range(j + 1, len(parent2) + 1):
            if parent2[j] not in croBird:
                croBird[i] = parent2[j]
                break

    return croBird

if __name__ == '__main__':
    total_dis = 0
    # 无人机参数
    CAPACITY = 100  # 无人机最大容量
    DISTABCE = 20  # 无人机最大行驶距离
    C0 = 5          # 无人机启动惩成本
    C1 = 1          #无人机单位距离的飞行成本
    C2 = 10         #无人机晚于配送时间的单位时间惩罚成本
    V = 60          # 无人机的速度

    # 其他参数
    bestfit = []  # 记录每代最优值
    DC = 6  # 配送中心个数
    C = 100  # 卸货点数量
    time_total = 0
    m = 10   #每个卸货点单次生成的订单量的最大值
    t = 180   # 生成订单的时间间隔
    customer = generate_coordinates(DC, C)
    print("配送中心：",customer[:DC])
    print("卸货点：",customer[DC:])

    while time_total <= 1440:
        birdNum = 40  # 粒子数量
        w = 0.2  # 惯性因子
        c1 = 0.4  # 自我认知因子
        c2 = 0.4  # 社会认知因子
        pBest, pLine = 0, []  # 当前最优值、当前最优解，（自我认知部分）
        gBest, gLine = 0, []  # 全局最优值、全局最优解，（社会认知部分）

        # 其他参数
        iterMax = 1000  # 迭代次数
        Customer, time, Demand = generate_order(customer,DC,m)

        dis_matrix = calDistance(Customer)  # 计算城市间距离

        # 分配卸货点到配送中心
        distribution_centers = assign_distribution_center(dis_matrix, DC, C)
        bestfit_list, gLine_list = [], []

        for certer_number in range(len(distribution_centers)):
            distribution_center = distribution_centers[certer_number]
            birdPop = [greedy(distribution_center, dis_matrix, certer_number) for i in range(birdNum)]  # 贪婪算法构造初始解
            birdPop_car, fits, dis_sum = calFitness(birdPop, certer_number, Demand, dis_matrix, CAPACITY, DISTABCE, C0, C1,C2,
                                           time,V)  # 分配无人机，计算种群适应度

            gBest = pBest = min(fits)  # 全局最优值、当前最优值
            gLine = pLine = birdPop[fits.index(min(fits))]  # 全局最优解、当前最优解
            gLine_car = pLine_car = birdPop_car[fits.index(min(fits))]

            iterI = 0  # 当前迭代次数
            while iterI <= iterMax:  # 迭代开始
                for i in range(birdNum):
                    birdPop[i] = crossover(birdPop[i], pLine, gLine, w, c1, c2)

                birdPop_car, fits,dis_sum = calFitness(birdPop, certer_number, Demand, dis_matrix, CAPACITY, DISTABCE, C0,
                                               C1,C2,time,V)  # 分配无人机，计算种群适应度
                pBest, pLine, pLine_car = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))]
                if min(fits) <= gBest:
                    gBest, gLine, gLine_car = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))]

                iterI += 1

            bestfit_list.append(gBest)
            gLine_list.append(gLine_car)

        print(gLine_list)  # 路径顺序
        for i in range(len(gLine_list)):
            print("第",i,"配送中心出动",len(gLine_list[i]))
        print("最优值：", sum(bestfit_list))
        total_dis += sum(bestfit_list)
        time_total += t
    print("总最优值",total_dis)
