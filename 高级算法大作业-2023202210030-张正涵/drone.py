# 配送中心和卸货点的位置
distribution_centers = [(0, 0), (10, 10)]  # 示例位置
drop_off_points = [(3, 4), (5, 6), (7, 8), (9, 10)]  # 示例位置

# 现在假定总体范围是10*10的网格图，请你把它们以可视化的方式展示出来
import matplotlib.pyplot as plt
import numpy as np

def show_net():
    plt.figure(figsize=(6, 6))
    plt.scatter([x for x, y in distribution_centers], [y for x, y in distribution_centers], c='r', marker='s', label='Distribution Center')
    plt.scatter([x for x, y in drop_off_points], [y for x, y in drop_off_points], c='b', marker='o', label='Drop Off Point')
    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.xticks(np.arange(-1, 11, 1))
    plt.yticks(np.arange(-1, 11, 1))
    plt.grid()
    plt.legend()
    plt.show()

# show_net()

orders = []
time_interval = 10  # 时间间隔，单位为分钟
num_intervals = 6  # 时间间隔的数量
import random
from pprint import pprint

def random_orders():
    for t in range(num_intervals):
        num_order = random.randint(0, len(drop_off_points))
        for _ in range(num_order):
            drop_off_point = random.choice(drop_off_points)
            priority = random.choice(["一般", "较紧急", "紧急"])
            orders.append({"time": t * time_interval, "drop_off_point": drop_off_point, "priority": priority})
    print(num_intervals*time_interval, "分钟内的订单：")
    pprint(orders)

from queue import PriorityQueue

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search_with_time(start, goal, graph, start_time, speed):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = (None, start_time)  # 记录前驱节点和到达时间
    cost_so_far[start] = 0

    while not frontier.empty():
        current_priority, current = frontier.get()
        current_time = came_from[current][1]  # 当前节点的到达时间

        if current == goal:
            break

        for next in graph[current]:
            travel_time = graph[current][next] / speed  # 计算飞行时间
            new_cost = cost_so_far[current] + graph[current][next]
            next_time = current_time + travel_time  # 计算到达下一个节点的时间

            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put((priority, next))
                came_from[next] = (current, next_time)

    return came_from, cost_so_far

# 重建路径，并包含时间信息
def reconstruct_path_with_time(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append((current, came_from[current][1]))
        current = came_from[current][0]
    path.append((start, came_from[start][1]))
    path.reverse()
    return path

random.seed(0)
random_orders()

graph = {
    (0, 0): {(3, 4): 5, (5, 6): 8, (7, 8): 10, (9, 10): 13},
    (3, 4): {(0, 0): 5, (5, 6): 3, (7, 8): 6, (9, 10): 9},
    (5, 6): {(0, 0): 8, (3, 4): 3, (7, 8): 3, (9, 10): 6},
    (7, 8): {(0, 0): 10, (3, 4): 6, (5, 6): 3, (9, 10): 3},
    (9, 10): {(0, 0): 13, (3, 4): 9, (5, 6): 6, (7, 8): 3, (10, 10): 1},
    (10, 10): {(9, 10): 1}
}

def schedule_and_plan_with_time(orders, distribution_centers, graph, max_load, max_distance, speed):
    # 按优先级排序订单
    priority_order = {"紧急": 1, "较紧急": 2, "一般": 3}
    orders.sort(key=lambda x: (priority_order[x["priority"]], x["time"]))

    print("按优先级排序的订单：")
    pprint(orders)
    
    routes = []
    
    # 送达时间限制
    priority_time_limit = {"紧急": 30, "较紧急": 90, "一般": 180}

    for current_time in range(0, time_interval * num_intervals, time_interval):
        current_orders = [order for order in orders if order["time"] <= current_time]
        orders = [order for order in orders if order not in current_orders]

        while current_orders:
            current_load = 0
            current_center = random.choice(distribution_centers)   # 随机挑选配送中心
            route = [(current_center, current_time)]
            total_distance = 0

            while current_load < max_load and current_orders:
                order = current_orders.pop(0)
                start = route[-1][0]
                start_time = route[-1][1]
                goal = order["drop_off_point"]
                delivery_deadline = order["time"] + priority_time_limit[order["priority"]]
                came_from, cost_so_far = a_star_search_with_time(start, goal, graph, start_time, speed)

                if total_distance + cost_so_far[goal] <= max_distance and came_from[goal][1] <= delivery_deadline:
                    path = reconstruct_path_with_time(came_from, start, goal)
                    route.extend(path[1:])  # 加入路径，去掉起点
                    total_distance += cost_so_far[goal]
                    current_load += 1
                else:
                    current_orders.insert(0, order)
                      # 放回订单队列
                    break
            if(len(route)>1):
                if route[-1][0] != current_center:
                    came_from, cost_so_far = a_star_search_with_time(route[-1][0], current_center, graph, route[-1][1], speed)
                    path_back = reconstruct_path_with_time(came_from, route[-1][0], current_center)
                    route.extend(path_back[1:])  # 加入返回路径，去掉起点

                routes.append(route)
    
    return routes

# 调度和路径规划
max_load = 2  # 无人机一次最多携带2个物品
max_distance = 20  # 无人机一次飞行最远路程为20公里
speed = 60 / 60  # 速度，单位公里/分钟
routes = schedule_and_plan_with_time(orders, distribution_centers, graph, max_load, max_distance, speed)
print("规划的路径（包含时间）:")
for route in routes:
    print(route)