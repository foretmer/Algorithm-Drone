from GeneticAlgorithm import *
from draw import *
from evaluate import *
import pandas as pd

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 1000)  # 设置打印宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行
pd.set_option('display.max_columns', None)  # 设置显示最大列



# 常量定义
DELIVERY_RADIUS = 20   # 无人机最大飞行距离 (公里)
DRONE_SPEED = 60       # 无人机速度 (公里/小时)
DRONE_CAPACITY = 3     # 无人机最大载货量
TIME_INTERVAL = 1     # 时间间隔
TIME_LIMITS = {1: 0.5, 2: 1.5, 3: 3.0}  # 订单优先级对应的时间限制 (小时)
DELAY_PENALTY = {1: 10, 2: 5, 3: 1}     # 订单优先级对应的延迟罚分


# 当前时间
current_time = 0
# 配送中心
centers = [(10, 10), (27, 10), (18, 25)]
# 卸货点
points = [
    [ (6, 8), (5, 15),  (8, 15), (9, 7), (10, 18), (12, 12), (12, 9), (15, 5), (17, 15), (19, 13),
     (5, 12), (12, 3), (17, 7), (2, 10), (8, 3), (17, 10), (10, 2), (12, 7), (9, 11), ],
    [(25, 2), (22, 10), (25, 18), (27, 13), (28, 7), (30, 12), (30, 5),  (33, 4), (34, 9), (32, 11),
     (33, 7), (21, 15), (32, 15), (28, 10), (24, 7),],
    [(18, 23), (17, 30), (20, 25), (24, 27), (13, 27), (14, 22), (19, 17), (20, 33), (20, 20), (25, 22), (10, 25)]
]

draw_graph(centers, points[0]+points[1]+points[2])



df = pd.DataFrame(columns=["时间","中心","订单数量", "配送数量", "延迟配送订单数量", "无人机数量", "超载数量", "超最长距离数量", "平均无人机使用效率"])
orders_delay = [None] * len(centers)

# 模拟24个时间步骤
while current_time < TIME_INTERVAL*24:
    print("\ncurrent_time: ", current_time, end='\n\n')
    best_route = [None] * len(centers)

    for i in range(len(centers)):
        print("\ncenter: ", centers[i], end='\n\n')

        # 生成订单 订单包含 订单ID、卸货点坐标、优先级、生成时间
        orders = generate_orders(points[i], current_time, count=23)

        # 将上一个时间步延迟的订单加入本次时间步的订单中,并提高优先级
        if orders_delay[i] is not None:
            for ord in orders_delay[i]:
                ord_new = (ord[0], ord[1], ord[2]-1, ord[3])
                orders.append(ord_new)
        print(f"orders of center{i}: ", orders)

        # 运行遗传算法。这里只运行了一遍遗传算法，应该运行多次防止出现偶然现象
        best_route[i], best_fitnesses = genetic_algorithm(orders, centers[i],
                                                          population_size=200,
                                                          generations=1000,
                                                          mutation_rate=0.01,
                                                          current_time=current_time)
        print(f"center{i}的最佳配送路径:", best_route[i])
        draw_Fitness_Over_Generations(best_fitnesses, current_time, centers[i])


        # 计算best_route中每个无人机的使用充分程度
        grade = grade_of_usage(best_route[i], centers[i])
        orders_delay[i], index = delay_orders(best_route[i], grade)
        print(f"orders_delay of center{i}: ", orders_delay[i])

        # 在best_route中删除orders_delay
        if index is not None:
            best_route[i].pop(index)

        # 评估
        if orders_delay[i] is not None:
            print(f"center{i}的最佳配送路径（延迟后）:", best_route[i])
        list_e = evalu(current_time, centers[i], orders, best_route[i], orders_delay[i], grade)
        df.loc[len(df)] = list_e

    # 绘制最优路线
    draw_graph_with_route(centers, points[0]+points[1]+points[2], best_route, current_time)
    current_time += TIME_INTERVAL

# 打印定量评估结果
print(df)