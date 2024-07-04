# Description: Python 调用 Gurobi 建模求解 VRPTW 问题
import time
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import *


class Data:
    customerNum = 0
    nodeNum = 0
    vehicleNum = 0
    capacity = 0
    corX = []
    corY = []
    demand = []
    serviceTime = []
    readyTime = []
    dueTime = []
    distanceMatrix = [[]]


def readData(path, customerNum):
    data = Data()
    data.customerNum = customerNum
    if customerNum is not None:
        data.nodeNum = customerNum + 6

    with open(path, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            count += 1
            if count == 5:
                line = line[:-1]
                s = re.split(r" +", line)
                data.vehicleNum = int(s[1])
                data.capacity = float(s[2])
            elif count >= 10 and (customerNum is None or count <= 12 + customerNum):
                line = line[:-1]
                s = re.split(r" +", line)
                data.corX.append(float(s[2]))
                data.corY.append(float(s[3]))
                data.demand.append(float(s[4]))
                data.readyTime.append(float(s[5]))
                data.dueTime.append(float(s[6]))
                data.serviceTime.append(float(s[7]))
    data.nodeNum = len(data.corX) + 3

    data.customerNum = data.nodeNum - 6


    # 回路
    data.corX.append(data.corX[0])
    data.corY.append(data.corY[0])
    data.demand.append(data.demand[0])
    data.readyTime.append(data.readyTime[0])
    data.dueTime.append(data.dueTime[0])
    data.serviceTime.append(data.serviceTime[0])
    data.corX.append(data.corX[1])
    data.corY.append(data.corY[1])
    data.demand.append(data.demand[1])
    data.readyTime.append(data.readyTime[1])
    data.dueTime.append(data.dueTime[1])
    data.serviceTime.append(data.serviceTime[1])
    data.corX.append(data.corX[2])
    data.corY.append(data.corY[2])
    data.demand.append(data.demand[2])
    data.readyTime.append(data.readyTime[2])
    data.dueTime.append(data.dueTime[2])
    data.serviceTime.append(data.serviceTime[2])

    # 计算距离矩阵
    data.distanceMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(data.nodeNum):
        for j in range(i + 1, data.nodeNum):
            distance = math.sqrt((data.corX[i] - data.corX[j]) ** 2 + (data.corY[i] - data.corY[j]) ** 2)
            data.distanceMatrix[i][j] = data.distanceMatrix[j][i] = distance
    return data


class Solution:
    ObjVal = 0
    X = [[]]
    S = [[]]
    routes = [[]]
    routeNum = 0

    def __init__(self, data, model):
        self.ObjVal = model.ObjVal
        # X_ijk
        self.X = [[([0] * data.vehicleNum) for _ in range(data.nodeNum)] for _ in range(data.nodeNum)]
        # S_ik
        self.S = [([0] * data.vehicleNum) for _ in range(data.nodeNum)]
        # routes
        self.routes = []


def getSolution(data, model):
    solution = Solution(data, model)  # 初始化 Solution 对象

    for m in model.getVars():  # 遍历所有变量
        split_arr = re.split(r"_", m.VarName)  # 分割变量名
        if split_arr[0] == 'X' and m.x > 0.5:  # 检查是否为 X_ijk 变量且其值大于 0.5
            solution.X[int(split_arr[1])][int(split_arr[2])][int(split_arr[3])] = m.x  # 更新 solution 中的 X 变量值
        elif split_arr[0] == 'S' and m.x > 0.5:  # 检查是否为 S_ik 变量且其值大于 0.5
            solution.S[int(split_arr[1])][int(split_arr[2])] = m.x  # 更新 solution 中的 S 变量值
    for k in range(data.vehicleNum):  # 遍历每辆车
        i = 0  # 初始节点（通常开始节点是0）
        subRoute = []  # 用于存储当前车辆路径的子路线
        subRoute.append(i)  # 将起点加入路径
        finish = False  # 设置结束标志，初始为False，表示路径尚未完成

        while not finish:  # 当路径未完成时，继续循环
            for j in range(data.nodeNum):  # 遍历所有节点，寻找下一个线路
                if solution.X[i][j][k] > 0.5:  # 如果车辆从i到j，是最优解中的一条路径
                    subRoute.append(j)  # 将节点j加入路径
                    i = j  # 更新当前节点为j
                    if j == data.nodeNum - 1:  # 如果到达终点节点，路径结束
                        finish = True

        if len(subRoute) >= 3:  # 确保路径至少经过了两个节点（起点和所有访问的节点）
            subRoute[-1] = 0  # 将子路径的最后一个节点变为起点（假设回到出发点）
            solution.routes.append(subRoute)  # 将该子路径加入solution.routes
            solution.routeNum += 1  # 更新路径数
    return solution
def getSolution0(data, model):
    solution = Solution(data, model)  # 初始化 Solution 对象

    for m in model.getVars():  # 遍历所有变量
        split_arr = re.split(r"_", m.VarName)  # 分割变量名
        if split_arr[0] == 'X' and m.x > 0.5:  # 检查是否为 X_ijk 变量且其值大于 0.5
            solution.X[int(split_arr[1])][int(split_arr[2])][int(split_arr[3])] = m.x  # 更新 solution 中的 X 变量值
        elif split_arr[0] == 'S' and m.x > 0.5:  # 检查是否为 S_ik 变量且其值大于 0.5
            solution.S[int(split_arr[1])][int(split_arr[2])] = m.x  # 更新 solution 中的 S 变量值
    for k in range(data.vehicleNum):  # 遍历每辆车
        i = 0  # 初始节点（通常开始节点是0）
        subRoute = []  # 用于存储当前车辆路径的子路线
        subRoute.append(i)  # 将起点加入路径
        finish = False  # 设置结束标志，初始为False，表示路径尚未完成

        while not finish:  # 当路径未完成时，继续循环
            for j in r0:  # 遍历所有节点，寻找下一个线路
                if solution.X[i][j][k] > 0.5:  # 如果车辆从i到j，是最优解中的一条路径
                    subRoute.append(j)  # 将节点j加入路径
                    i = j  # 更新当前节点为j
                    if j == r0[-1]:  # 如果到达终点节点，路径结束
                        finish = True

        if len(subRoute) >= 3:  # 确保路径至少经过了两个节点（起点和所有访问的节点）
            subRoute[-1] = 0  # 将子路径的最后一个节点变为起点（假设回到出发点）
            solution.routes.append(subRoute)  # 将该子路径加入solution.routes
            solution.routeNum += 1  # 更新路径数
    return solution
def getSolution1(data, model):
    solution = Solution(data, model)  # 初始化 Solution 对象

    for m in model.getVars():  # 遍历所有变量
        split_arr = re.split(r"_", m.VarName)  # 分割变量名
        if split_arr[0] == 'X' and m.x > 0.5:  # 检查是否为 X_ijk 变量且其值大于 0.5
            solution.X[int(split_arr[1])][int(split_arr[2])][int(split_arr[3])] = m.x  # 更新 solution 中的 X 变量值
        elif split_arr[0] == 'S' and m.x > 0.5:  # 检查是否为 S_ik 变量且其值大于 0.5
            solution.S[int(split_arr[1])][int(split_arr[2])] = m.x  # 更新 solution 中的 S 变量值
    for k in range(data.vehicleNum):  # 遍历每辆车
        i = 1  # 初始节点（通常开始节点是0）
        subRoute = []  # 用于存储当前车辆路径的子路线
        subRoute.append(i)  # 将起点加入路径
        finish = False  # 设置结束标志，初始为False，表示路径尚未完成

        while not finish:  # 当路径未完成时，继续循环
            for j in r1:  # 遍历所有节点，寻找下一个线路
                if solution.X[i][j][k] > 0.5:  # 如果车辆从i到j，是最优解中的一条路径
                    subRoute.append(j)  # 将节点j加入路径
                    i = j  # 更新当前节点为j
                    if j == r1[-1]:  # 如果到达终点节点，路径结束
                        finish = True

        if len(subRoute) >= 3:  # 确保路径至少经过了两个节点（起点和所有访问的节点）
            subRoute[-1] = 1  # 将子路径的最后一个节点变为起点（假设回到出发点）
            solution.routes.append(subRoute)  # 将该子路径加入solution.routes
            solution.routeNum += 1  # 更新路径数
    return solution

def getSolution2(data, model):
    solution = Solution(data, model)  # 初始化 Solution 对象

    for m in model.getVars():  # 遍历所有变量
        split_arr = re.split(r"_", m.VarName)  # 分割变量名
        if split_arr[0] == 'X' and m.x > 0.5:  # 检查是否为 X_ijk 变量且其值大于 0.5
            solution.X[int(split_arr[1])][int(split_arr[2])][int(split_arr[3])] = m.x  # 更新 solution 中的 X 变量值
        elif split_arr[0] == 'S' and m.x > 0.5:  # 检查是否为 S_ik 变量且其值大于 0.5
            solution.S[int(split_arr[1])][int(split_arr[2])] = m.x  # 更新 solution 中的 S 变量值
    for k in range(data.vehicleNum):  # 遍历每辆车
        i = 2  # 初始节点（通常开始节点是0）
        subRoute = []  # 用于存储当前车辆路径的子路线
        subRoute.append(i)  # 将起点加入路径
        finish = False  # 设置结束标志，初始为False，表示路径尚未完成

        while not finish:  # 当路径未完成时，继续循环
            for j in r2:  # 遍历所有节点，寻找下一个线路
                if solution.X[i][j][k] > 0.5:  # 如果车辆从i到j，是最优解中的一条路径
                    subRoute.append(j)  # 将节点j加入路径
                    i = j  # 更新当前节点为j
                    if j == r2[-1]:  # 如果到达终点节点，路径结束
                        finish = True

        if len(subRoute) >= 3:  # 确保路径至少经过了两个节点（起点和所有访问的节点）
            subRoute[-1] = 2  # 将子路径的最后一个节点变为起点（假设回到出发点）
            solution.routes.append(subRoute)  # 将该子路径加入solution.routes
            solution.routeNum += 1  # 更新路径数
    return solution
def plot_solution(solution, customer_num):
    # 设置绘图框的标题和坐标标签
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{customer_num} Customers")  # 标题显示数据类型和顾客数量

    # 绘制起点
    plt.scatter(data.corX[0], data.corY[0], c='blue', alpha=1, marker=',', linewidths=3, label='depot')  # 起点
    plt.scatter(data.corX[1], data.corY[1], c='blue', alpha=1, marker=',', linewidths=3, label='depot')  # 起点
    plt.scatter(data.corX[2], data.corY[2], c='blue', alpha=1, marker=',', linewidths=3, label='depot')  # 起点

    # 绘制客户点
    plt.scatter(data.corX[3:-3], data.corY[3:-3], c='red', alpha=1, marker='o', linewidths=3,
                label='customer')  # 普通站点

    # 遍历每个路径，并绘制路径上的每一段弧
    for k in range(solution.routeNum):
        for i in range(len(solution.routes[k]) - 1):
            a = solution.routes[k][i]
            b = solution.routes[k][i + 1]
            x = [data.corX[a], data.corX[b]]
            y = [data.corY[a], data.corY[b]]
            plt.plot(x, y, 'k', linewidth=1)  # 使用黑色线绘制路径上的线段

    # 绘制图表
    plt.grid(False)  # 隐藏网格线
    # plt.legend(loc='best')  # 添加图例



def print_solution(solution, data):
    for index, subRoute in enumerate(solution.routes):
        distance = 0
        load = 0
        for i in range(len(subRoute) - 1):
            distance += data.distanceMatrix[subRoute[i]][subRoute[i + 1]]
            load += data.demand[subRoute[i]]
        print(f"Route-{index + 1} : {subRoute} , distance: {distance} , load: {load}")


def solve(data):
    # 声明模型
    model = Model("VRPTW")
    # 模型设置
    # 关闭输出
    model.setParam('OutputFlag', 0)
    # 定义变量
    X = [[[[] for _ in range(data.vehicleNum)] for _ in range(data.nodeNum)] for _ in range(data.nodeNum)]
    S = [[[] for _ in range(data.vehicleNum)] for _ in range(data.nodeNum)]
    for i in range(data.nodeNum):
        for k in range(data.vehicleNum):
            S[i][k] = model.addVar(data.readyTime[i], data.dueTime[i], vtype=GRB.CONTINUOUS, name=f'S_{i}_{k}')
            for j in range(data.nodeNum):
                X[i][j][k] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}_{k}")
    # 目标函数
    obj = LinExpr(0)
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            if i != j:
                for k in range(data.vehicleNum):
                    obj.addTerms(data.distanceMatrix[i][j], X[i][j][k])
    model.setObjective(obj, GRB.MINIMIZE)
    # 约束1：车辆只能从一个点到另一个点
    for i in range(1, data.nodeNum - 1):
        expr = LinExpr(0)
        for j in range(data.nodeNum):
            if i != j:
                for k in range(data.vehicleNum):
                    if i != 0 and i != data.nodeNum - 1:
                        expr.addTerms(1, X[i][j][k])
        model.addConstr(expr == 1)
    # 约束2：车辆必须从仓库出发
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for j in range(1, data.nodeNum):
            expr.addTerms(1, X[0][j][k])
        model.addConstr(expr == 1)
    # 约束3：车辆经过一个点就必须离开一个点
    for k in range(data.vehicleNum):
        for h in range(1, data.nodeNum - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for i in range(data.nodeNum):
                if h != i:
                    expr1.addTerms(1, X[i][h][k])
            for j in range(data.nodeNum):
                if h != j:
                    expr2.addTerms(1, X[h][j][k])
            model.addConstr(expr1 == expr2)
    # 约束4：车辆最终返回仓库
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for i in range(data.nodeNum - 1):
            expr.addTerms(1, X[i][data.nodeNum - 1][k])
        model.addConstr(expr == 1)
    # 约束5：车辆容量约束
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for i in range(1, data.nodeNum - 1):
            for j in range(data.nodeNum):
                if i != 0 and i != data.nodeNum - 1 and i != j:
                    expr.addTerms(data.demand[i], X[i][j][k])
        model.addConstr(expr <= data.capacity)
    # 约束6：时间窗约束
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                if i != j:
                    model.addConstr(S[i][k] + data.distanceMatrix[i][j] - S[j][k] <= M - M * X[i][j][k])
    # 记录求解开始时间
    start_time = time.time()
    # 求解
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("-" * 20, "Solved Successfully", '-' * 20)
        # 输出求解总用时
        print(f"Solve Time: {time.time() - start_time} s")
        print(f"Total Travel Distance: {model.ObjVal}")
        solution = getSolution(data, model)
        plot_solution(solution, data.customerNum)
        print_solution(solution, data)
    else:
        print("此题无解")
def solve0(data):
    # 声明模型
    model = Model("VRPTW0")
    # 模型设置
    # 关闭输出
    model.setParam('OutputFlag', 0)
    # 定义变量
    X = [[[[] for _ in range(data.vehicleNum)] for _ in range(data.nodeNum)] for _ in range(data.nodeNum)]
    S = [[[] for _ in range(data.vehicleNum)] for _ in range(data.nodeNum)]
    for i in r0:
        for k in range(data.vehicleNum):
            S[i][k] = model.addVar(data.readyTime[i], data.dueTime[i], vtype=GRB.CONTINUOUS, name=f'S_{i}_{k}')
            for j in r0:
                X[i][j][k] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}_{k}")
    # 目标函数
    obj = LinExpr(0)
    for i in r0:
        for j in r0:
            if i != j:
                for k in range(data.vehicleNum):
                    obj.addTerms(data.distanceMatrix[i][j], X[i][j][k])
    model.setObjective(obj, GRB.MINIMIZE)
    # 约束1：车辆只能从一个点到另一个点
    for i in r0[1:-1]:
        expr = LinExpr(0)
        for j in r0:
            if i != j:
                for k in range(data.vehicleNum):
                    if i != 0 and i != len(r0) - 1:
                        expr.addTerms(1, X[i][j][k])
        model.addConstr(expr == 1)
    # 约束2：车辆必须从仓库出发
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for j in r0[1:]:
            expr.addTerms(1, X[0][j][k])
        model.addConstr(expr == 1)
    # 约束3：车辆经过一个点就必须离开一个点
    for k in range(data.vehicleNum):
        for h in r0[1:-1]:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for i in r0:
                if h != i:
                    expr1.addTerms(1, X[i][h][k])
            for j in r0:
                if h != j:
                    expr2.addTerms(1, X[h][j][k])
            model.addConstr(expr1 == expr2)
    # 约束4：车辆最终返回仓库
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for i in r0[:-1]:
            expr.addTerms(1, X[i][r0[-1]][k])
        model.addConstr(expr == 1)
    # 约束5：车辆容量约束
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for i in r0[1:-1]:
            for j in r0:
                if i != 0 and i != data.nodeNum - 1 and i != j:
                    expr.addTerms(data.demand[i], X[i][j][k])
        model.addConstr(expr <= data.capacity)
    # 约束6：时间窗约束
    for k in range(data.vehicleNum):
        for i in r0:
            for j in r0:
                if i != j:
                    model.addConstr(S[i][k] + data.distanceMatrix[i][j] - S[j][k] <= M - M * X[i][j][k])
    # 约束7：
    # 为每个车辆k添加约束条件
    for k in range(data.vehicleNum):
        for i in r0:
            for j in r0:
                 if data.distanceMatrix[i][j] < 20:
                    # 只有当弧ij的长度小于20时，才需要添加约束
                    model.addConstr(X[i][j][k] <= 1)
                 else:
                    # 如果弧ij的长度大于或等于20，车辆k不能选择这条弧
                    model.addConstr(X[i][j][k] == 0)

    # 记录求解开始时间
    start_time = time.time()
    # 求解
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("-" * 20, "配送中心0 Solved Successfully", '-' * 20)
        # 输出求解总用时
        print(f"Solve Time: {time.time() - start_time} s")
        print(f"Total Travel Distance: {model.ObjVal}")
        solution = getSolution0(data, model)
        print(solution.routes)
        plot_solution(solution, data.customerNum)
        print_solution(solution, data)
    else:
        print("此题无解")
def solve1(data):
    # 声明模型
    model = Model("VRPTW1")
    # 模型设置
    # 关闭输出
    model.setParam('OutputFlag', 0)
    # 定义变量
    X = [[[[] for _ in range(data.vehicleNum)] for _ in range(data.nodeNum)] for _ in range(data.nodeNum)]
    S = [[[] for _ in range(data.vehicleNum)] for _ in range(data.nodeNum)]
    for i in r1:
        for k in range(data.vehicleNum):
            S[i][k] = model.addVar(data.readyTime[i], data.dueTime[i], vtype=GRB.CONTINUOUS, name=f'S_{i}_{k}')
            for j in r1:
                X[i][j][k] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}_{k}")
    # 目标函数
    obj = LinExpr(0)
    for i in r1:
        for j in r1:
            if i != j:
                for k in range(data.vehicleNum):
                    obj.addTerms(data.distanceMatrix[i][j], X[i][j][k])
    model.setObjective(obj, GRB.MINIMIZE)
    # 约束1：车辆只能从一个点到另一个点
    for i in r1[1:-1]:
        expr = LinExpr(0)
        for j in r1:
            if i != j:
                for k in range(data.vehicleNum):
                    if i != 0 and i != len(r1) - 1:
                        expr.addTerms(1, X[i][j][k])
        model.addConstr(expr == 1)
    # 约束2：车辆必须从仓库出发
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for j in r1[1:]:
            expr.addTerms(1, X[1][j][k])
        model.addConstr(expr == 1)
    # 约束3：车辆经过一个点就必须离开一个点
    for k in range(data.vehicleNum):
        for h in r1[1:-1]:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for i in r1:
                if h != i:
                    expr1.addTerms(1, X[i][h][k])
            for j in r1:
                if h != j:
                    expr2.addTerms(1, X[h][j][k])
            model.addConstr(expr1 == expr2)
    # 约束4：车辆最终返回仓库
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for i in r1[:-1]:
            expr.addTerms(1, X[i][r1[-1]][k])
        model.addConstr(expr == 1)
    # 约束5：车辆容量约束
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for i in r1[1:-1]:
            for j in r1:
                if i != 0 and i != data.nodeNum - 1 and i != j:
                    expr.addTerms(data.demand[i], X[i][j][k])
        model.addConstr(expr <= data.capacity)
    # 约束6：时间窗约束
    for k in range(data.vehicleNum):
        for i in r1:
            for j in r1:
                if i != j:
                    model.addConstr(S[i][k] + data.distanceMatrix[i][j] - S[j][k] <= M - M * X[i][j][k])
    # 约束7：
    # 为每个车辆k添加约束条件
    for k in range(data.vehicleNum):
        for i in r1:
            for j in r1:
                if data.distanceMatrix[i][j] < 20:
                    # 只有当弧ij的长度小于20时，才需要添加约束
                    model.addConstr(X[i][j][k] <= 1)
                else:
                    # 如果弧ij的长度大于或等于20，车辆k不能选择这条弧
                    model.addConstr(X[i][j][k] == 0)
    # 记录求解开始时间
    start_time = time.time()
    # 求解
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("-" * 20, "配送中心1 Solved Successfully", '-' * 20)
        # 输出求解总用时
        print(f"Solve Time: {time.time() - start_time} s")
        print(f"Total Travel Distance: {model.ObjVal}")
        solution = getSolution1(data, model)
        print(solution.routes)
        plot_solution(solution, data.customerNum)
        print_solution(solution, data)
    else:
        print("此题无解")
def solve2(data):
    # 声明模型
    model = Model("VRPTW2")
    # 模型设置
    # 关闭输出
    model.setParam('OutputFlag', 0)
    # 定义变量
    X = [[[[] for _ in range(data.vehicleNum)] for _ in range(data.nodeNum)] for _ in range(data.nodeNum)]
    S = [[[] for _ in range(data.vehicleNum)] for _ in range(data.nodeNum)]
    for i in r2:
        for k in range(data.vehicleNum):
            S[i][k] = model.addVar(data.readyTime[i], data.dueTime[i], vtype=GRB.CONTINUOUS, name=f'S_{i}_{k}')
            for j in r2:
                X[i][j][k] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}_{k}")
    # 目标函数
    obj = LinExpr(0)
    for i in r2:
        for j in r2:
            if i != j:
                for k in range(data.vehicleNum):
                    obj.addTerms(data.distanceMatrix[i][j], X[i][j][k])
    model.setObjective(obj, GRB.MINIMIZE)
    # 约束1：车辆只能从一个点到另一个点
    for i in r2[1:-1]:
        expr = LinExpr(0)
        for j in r2:
            if i != j:
                for k in range(data.vehicleNum):
                    if i != 0 and i != len(r2) - 1:
                        expr.addTerms(1, X[i][j][k])
        model.addConstr(expr == 1)
    # 约束2：车辆必须从仓库出发
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for j in r2[1:]:
            expr.addTerms(1, X[2][j][k])
        model.addConstr(expr == 1)
    # 约束3：车辆经过一个点就必须离开一个点
    for k in range(data.vehicleNum):
        for h in r2[1:-1]:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for i in r2:
                if h != i:
                    expr1.addTerms(1, X[i][h][k])
            for j in r2:
                if h != j:
                    expr2.addTerms(1, X[h][j][k])
            model.addConstr(expr1 == expr2)
    # 约束4：车辆最终返回仓库
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for i in r2[:-1]:
            expr.addTerms(1, X[i][r2[-1]][k])
        model.addConstr(expr == 1)
    # 约束5：车辆容量约束
    for k in range(data.vehicleNum):
        expr = LinExpr(0)
        for i in r2[1:-1]:
            for j in r2:
                if i != 0 and i != data.nodeNum - 1 and i != j:
                    expr.addTerms(data.demand[i], X[i][j][k])
        model.addConstr(expr <= data.capacity)
    # 约束6：时间窗约束
    for k in range(data.vehicleNum):
        for i in r2:
            for j in r2:
                if i != j:
                    model.addConstr(S[i][k] + data.distanceMatrix[i][j] - S[j][k] <= M - M * X[i][j][k])
    # 约束7：
    # 为每个车辆k添加约束条件
    for k in range(data.vehicleNum):
        for i in r2:
            for j in r2:
                if data.distanceMatrix[i][j] < 20:
                    # 只有当弧ij的长度小于20时，才需要添加约束
                    model.addConstr(X[i][j][k] <= 1)
                else:
                    # 如果弧ij的长度大于或等于20，车辆k不能选择这条弧
                    model.addConstr(X[i][j][k] == 0)

    # 记录求解开始时间
    start_time = time.time()
    # 求解
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("-" * 20, "配送中心2 Solved Successfully", '-' * 20)
        # 输出求解总用时
        print(f"Solve Time: {time.time() - start_time} s")
        print(f"Total Travel Distance: {model.ObjVal}")
        solution = getSolution2(data, model)
        print(solution.routes)
        plot_solution(solution, data.customerNum)
        print_solution(solution, data)
    else:
        print("此题无解")





if __name__ == '__main__':
    r0 = []
    r1 = []
    r2 = []

    # 数据集路径
    data_path = '../spots（问题实例数据）.txt'
    # 顾客个数设置（从上往下读取完 customerNum 个顾客为止，例如c101文件中有100个顾客点，
    # 但是跑100个顾客点太耗时了，设置这个数是为了只选取一部分顾客点进行计算，用来快速测试算法）
    # 如果想用完整的顾客点进行计算，设置为None即可
    customerNum = 24


    # 一个很大的正数
    M = 10000000
    # 读取数据
    data = readData(data_path, customerNum)

    r0.append(0)
    r1.append(1)
    r2.append(2)
    for i in range(customerNum):
        if data.distanceMatrix[0][3+i]<=data.distanceMatrix[1][3+i] and data.distanceMatrix[0][3+i]<=data.distanceMatrix[2][3+i]:
            r0.append(3+i)
        elif data.distanceMatrix[1][3+i]<=data.distanceMatrix[0][3+i] and data.distanceMatrix[1][3+i]<=data.distanceMatrix[2][3+i]:
            r1.append(3+i)
        elif data.distanceMatrix[2][3+i]<=data.distanceMatrix[0][3+i] and data.distanceMatrix[2][3+i]<=data.distanceMatrix[1][3+i]:
            r2.append(3+i)
    r0.append(data.nodeNum-3)
    r1.append(data.nodeNum-2)
    r2.append(data.nodeNum-1)
    print(f"配送中心0:{r0}")
    print(f"配送中心1:{r1}")
    print(f"配送中心2:{r2}")


    # 输出相关数据
    print("-" * 20, "Problem Information", '-' * 20)
    print(f'Node Num: {data.nodeNum}')
    print(f'Customer Num: {data.customerNum}')
    print(f'Vehicle Num: {data.vehicleNum}')
    print(f'Vehicle Capacity: {data.capacity}')
    # 建模求解
    solve0(data)
    solve1(data)
    solve2(data)
    plt.show()  # 显示图表
