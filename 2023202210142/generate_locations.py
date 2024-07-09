import numpy as np
import matplotlib.pyplot as plt
# 随机生成配送中心和卸货点坐标
np.random.seed(800) # 520, 2000过程中有比较好的收敛趋势，600和6每个区域区分明显，6000还可以但是有角落样本集中，800还可以

# 订单优先级和对应的配送时间约束
priority_levels = {
    0: 180,  # 一般：3小时（180分钟）内配送到
    1: 90,   # 较紧急：1.5小时（90分钟）内配送到
    2: 30    # 紧急：0.5小时（30分钟）内配送到
}

def generate_map(map_size=25, num_distribution_centers=3, num_delivery_points=10, max_distance=10):
    # 生成配送中心坐标
    distribution_centers = np.random.randint(0, map_size, size=(num_distribution_centers, 2))

    # 生成卸货点坐标，确保每个卸货点距离最近的配送中心不超过max_distance且不重复
    delivery_points = []
    while len(delivery_points) < num_delivery_points:
        center = distribution_centers[np.random.randint(0, num_distribution_centers)]
        point = center + np.random.randint(-max_distance, max_distance + 1, size=2)
        point = np.clip(point, 0, map_size - 1)
        if np.any(np.linalg.norm(distribution_centers - point, axis=1) <= max_distance):
            if not any(np.array_equal(point, dp) for dp in delivery_points) and \
                    not any(np.array_equal(point, dc) for dc in distribution_centers):
                delivery_points.append(point)

    delivery_points = np.array(delivery_points)
    return distribution_centers, delivery_points

# 生成随机订单
def generate_orders(k, max_orders_per_interval):
    """
    生成随机订单，每个卸货点随机生成0到max_orders_per_interval个订单。
    每个订单包含配送时间约束和优先级信息。
    # 示例：生成当前时间间隔的订单
        orders = generate_orders(k, m)
        print("Generated orders (delivery point, priority, time constraint):")
        print(orders) [(0, 1, 90), (2, 0, 180), (4, 2, 30), (4, 0, 180), (7, 1, 90), (8, 2, 30), (9, 0, 180)]
    """
    orders = []
    for i in range(k):
        num_orders = np.random.randint(0, max_orders_per_interval + 1)
        for _ in range(num_orders):
            priority = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            time_constraint = priority_levels[priority]
            orders.append((i, priority, time_constraint))
    return orders

