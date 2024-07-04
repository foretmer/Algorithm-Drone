import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.approximation import traveling_salesman_problem


class Order:
    """
    订单类
    order_id: 订单id
    deadline: 订单需要在截止时间前送达
    priority: 订单优先级
    * 0: 紧急, 0.5小时内配送到
    * 1: 较紧急, 1.5小时内配送到
    * 2: 一般, 3小时内配送到即可
    point_id: 订单目的地
    """
    times = [30, 90, 180] # 静态变量，优先级对应的时间

    def __init__(self, current_time, order_id, priority, point_id):
        self.order_id = order_id  # 分配唯一的订单ID
        self.deadline = current_time + Order.times[priority]
        self.priority = priority
        self.point_id = point_id


def simulate_order_generation(points, minute):
    """
    模拟订单生成。
    :param points: 需要生成订单的点的列表
    :param minute: 订单生成的时刻（也就是current_time）
    :return: 一个包含所有生成订单的列表
    """
    random.seed()  # 初始化随机数生成器
    orders = []
    for point in points:
        for i in range(random.randint(1, 5)):  # 随机生成1到5个订单
            new_order = Order(minute, f"order_{minute}_{i}", random.randint(0, 2), point)
            orders.append(new_order)
    return orders


def cal_distance(point1, point2):
    """
    计算两个点之间的欧几里得距离，这样能保证满足三角不等式
    :param point1: 第一个点的坐标，形式为(x, y)的元组
    :param point2: 第二个点的坐标，形式为(x, y)的元组
    :return 两个点之间的欧几里得距离，取整数
    """
    distance = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    return round(distance)


def init_graph(centers, points):
    """
   初始化图结构，并根据给定的中心点和点集构建图。
   :param centers: 配送中心的字典，{中心标识: 中心坐标}
   :param points: 卸货点的字典，{点标识: 点坐标}
   :return G: 构建好的图
   """
    # 创建一个空的无向图
    G = nx.Graph()

    # 添加配送中心和卸货点节点
    for node, pos in centers.items():
        G.add_node(node, pos=pos)
    for node, pos in points.items():
        G.add_node(node, pos=pos)

    # 添加配送中心和卸货点之间的边（假设所有配送中心都可以向所有卸货点配送货物）
    for center in centers:
        for point in points:
            # 计算配送中心到卸货点的距离
            distance = cal_distance(centers[center], points[point])
            if distance < 10:
                G.add_edge(center, point, weight=distance)

    for point1 in points:
        for point2 in points:
            if point1 != point2:
                # 计算卸货点到卸货点的距离
                distance = cal_distance(points[point1], points[point2])
                if distance < 10:
                    G.add_edge(point1, point2, weight=distance)

    # 绘制图
    # if plt.get_fignums():
    #     plt.close('all')  # 关闭所有打开的图像，避免干扰
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
    #
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # plt.show()

    return G


# 决策函数
def make_decisions(G, orders, current_time, max_flight_distance=20, drone_capacity=3):
    """
    根据给定的图形G、订单列表和无人机参数，决定无人机的分配和飞行路线。
    :param G: 图形，表示配送区域中配送中心和卸货点之间的关系
    :param orders: 订单列表，每个订单包含截止时间、优先级和配送点信息
    :param max_flight_distance: 无人机的最大飞行距离，默认为20
    :param drone_capacity: 无人机的载货量，默认为3
    :param current_time: 当前时间
    :return drone_assignments: 无人机分配字典，键为配送中心，值为分配给该中心的订单列表。
    :return reserved_orders: 由于超出了无人机能力而保留的订单列表。
    """
    # 初始化无人机的分配
    drone_assignments = {center: [] for center in G.nodes() if "s" in center}
    reserved_orders = []

    # 对订单按照时效和优先级排序
    orders.sort(key=lambda x: (x.deadline - current_time, x.priority))

    # 分配订单到最近的配送中心
    for order in orders:
        shortest_distance = float('inf')
        selected_center = None
        for center in G.nodes():
            if "s" in center:
                if G.has_edge(center, order.point_id):
                    distance = G[center][order.point_id]["weight"]
                    if distance < shortest_distance:
                        shortest_distance = distance
                        selected_center = center
        drone_assignments[selected_center].append(order)

    # 根据无人机容量和订单数量决定派出多少架无人机
    for center, orders in drone_assignments.items():
        num_drones = (len(orders) + drone_capacity - 1) // drone_capacity  # 向上取整
        drone_routes = []

        # 为每架无人机规划路线
        for i in range(num_drones):
            drone_orders = orders[i * drone_capacity:(i + 1) * drone_capacity]
            # 创建一个子图，只包含配送中心和需要配送的卸货点
            sub_G = G.copy()
            for node in list(sub_G.nodes()):
                if node not in [center] + [order.point_id for order in drone_orders]:
                    sub_G.remove_node(node)

            # 使用Christofides算法找到近似最优路径
            tsp_path = traveling_salesman_problem(sub_G, cycle=True)
            tsp_path_length = sum(sub_G[tsp_path[i]][tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
            if tsp_path_length > max_flight_distance:
                # 如果路径长度超过最大飞行距离，则删除一个优先级最小的订单并重新规划路径
                orders.sort(key=lambda x: (x.deadline - current_time, x.priority), reverse=True)
                reserved_orders.append(orders.pop())
                tsp_path = traveling_salesman_problem(sub_G, cycle=True)
                tsp_path_length = sum(sub_G[tsp_path[i]][tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
            drone_routes.append((drone_orders, tsp_path, tsp_path_length))

        print(f"配送中心 {center} 将派出 {num_drones} 架无人机。")
        idx = 1
        for drone_orders, tsp_path, tsp_path_length in drone_routes:
            print(
                f"无人机{idx}将配送以下订单：{', '.join([order.order_id for order in drone_orders])}\n"
                f"路径为：{tsp_path}，路径长度为：{tsp_path_length}km")
            idx += 1

    return drone_assignments, reserved_orders


def main():
    # 定义配送中心和卸货点的位置
    centers = {
        "s0": (0, 0),
        "s1": (0, 3),
        "s2": (3, 0)
    }

    points = {
        "d0": (-4, 2),
        "d1": (5, -2),
        "d2": (2, 7),
        "d3": (-4, 4),
        "d4": (5, 6),
        "d5": (-6, -4)
    }

    G = init_graph(centers, points)

    # 模拟每隔10分钟生成订单并进行决策
    for minute in range(0, 180, 10):
        # 生成订单
        orders = simulate_order_generation(points, minute)

        print(f"====================第{minute}分钟：====================")
        # 进行决策
        drone_assignments, reversed_orders = make_decisions(G, orders, current_time=minute)
        orders += reversed_orders

        # 打印决策结果
        print("")
        for center, orders in drone_assignments.items():
            if len(orders) == 0:
                continue
            print(f"配送中心 {center} 的无人机将完成以下订单：")
            for order in orders:
                print(f"订单ID: {order.order_id}, d: {order.point_id}, 优先级: {order.priority}")

        # 等待
        # import time
        # time.sleep(10)


if __name__ == "__main__":
    main()
