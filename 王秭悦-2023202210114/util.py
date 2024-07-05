import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import art3d
import numpy as np
import random


# 常量定义
DELIVERY_RADIUS = 20   # 无人机最大飞行距离 (公里)
DRONE_SPEED = 60       # 无人机速度 (公里/小时)
DRONE_CAPACITY = 3     # 无人机最大载货量
TIME_INTERVAL = 1     # 时间间隔
TIME_LIMITS = {1: 0.5, 2: 1.5, 3: 3.0}  # 订单优先级对应的时间限制 (小时)
DELAY_PENALTY = {1: 10, 2: 5, 3: 1}     # 订单优先级对应的延迟罚分



# 无人机配送路径距离计算函数
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# 随机生成订单
def generate_orders(points, time, count):
    new_orders = []
    for i in range(random.randint(0, count)):
    # for i in range(count):
        priority = random.choice([1, 2, 3])
        point = random.choice(points)
        new_orders.append((i, point, priority, time))
    return new_orders


# 无人机使用效率分数
def grade_of_usage(individual, center):
    grade = []  # 存放某个个体中每条路径的无人机使用充分程度
    for route in individual:

        previous_point = center
        route_distance = 0
        priority_num = 0

        for order in route:
            i, point, priority, order_time = order

            # 计算某架无人机的飞行距离
            distance = calculate_distance(previous_point, point)
            route_distance += distance

            # 计算某架无人机高优先级货物数量
            if priority == 1:
                priority_num += 1
            elif priority == 2:
                priority_num += 0.5

            previous_point = point

        distance = calculate_distance(previous_point, center)
        route_distance += distance

        # 计算某架无人机实际载货数量
        cargo_num = len(route)
        grade_route = {}
        if cargo_num != 0:
            grade_route['cargo_num'] = cargo_num
            grade_route['route_distance'] = route_distance
            grade_route['priority_num'] = priority_num
            grade_route['grade_route'] = (grade_route['cargo_num'] / DRONE_CAPACITY
                                          + grade_route['route_distance'] / DELIVERY_RADIUS
                                          + 2*grade_route['priority_num'] / cargo_num)/4
        else:
            grade_route['cargo_num'] = 0
            grade_route['route_distance'] = 0
            grade_route['priority_num'] = 0
            grade_route['grade_route'] = 0

        grade.append(grade_route)
    return grade


def delay_orders(individual, grade_individual):
    min_grade = min(grade_individual, key=lambda d: d['grade_route'])
    min_index = grade_individual.index(min_grade)
    min_grade_route = individual[min_index]

    if len(min_grade_route) == 4:
        return None, None
    for order in min_grade_route:
        if order[2] == 1:
            return None, None
    return min_grade_route, min_index



def merge_lists(list):
    merged_list = []
    for i in list:
        merged_list.extend(i)
    return merged_list


def unmerge_lists(merged_list, Unmerged_list):
    nested_list = []
    current_index = 0
    for i in range(len(Unmerged_list)):
        nested_list.append(merged_list[current_index:current_index + len(Unmerged_list[i])])
        current_index += len(Unmerged_list[i])
    return nested_list
