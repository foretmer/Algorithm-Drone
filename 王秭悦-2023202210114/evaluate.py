import pandas as pd

# 常量定义
DELIVERY_RADIUS = 20   # 无人机最大飞行距离 (公里)
DRONE_SPEED = 60       # 无人机速度 (公里/小时)
DRONE_CAPACITY = 3     # 无人机最大载货量
TIME_INTERVAL = 1      # 时间间隔
TIME_LIMITS = {1: 0.5, 2: 1.5, 3: 3.0}  # 订单优先级对应的时间限制 (小时)
DELAY_PENALTY = {1: 10, 2: 5, 3: 1}     # 订单优先级对应的延迟罚分


def evalu(time, center, orders, best_route, orders_delay, grade):
    df_list=[]
    df_list.append(time)
    df_list.append(center)
    df_list.append(len(orders))
    df_list.append(sum(len(route) for route in best_route))
    df_list.append(len(orders_delay) if orders_delay else 0)
    df_list.append(len(best_route))
    overload_num = sum(1 for g in grade if g['cargo_num']>DRONE_CAPACITY)
    df_list.append(overload_num)
    overdistance_num = sum(1 for g in grade if g['route_distance']>DELIVERY_RADIUS)
    df_list.append(overdistance_num)
    average_grade = sum(g['grade_route'] for g in grade) / len(grade)
    df_list.append(average_grade)

    return df_list