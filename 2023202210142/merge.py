import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def aggregate_points(distribution_centers, delivery_points, edge_threshold=0.5):
    """
    对客户点进行聚合优化，将客户点分配到最合适的配送中心。

    参数:
    - distribution_centers: 配送中心的坐标数组，形状为 (j, 2)
    - delivery_points: 客户点的坐标数组，形状为 (k, 2)
    - edge_threshold: 边缘系数阈值，默认为 1.5

    返回:
    - point_assignments: 每个客户点所属的配送中心索引数组
    """
    # STEP1: 计算客户点与配送中心的距离矩阵
    distance_matrix = cdist(delivery_points, distribution_centers)

    # STEP2: 计算边缘系数
    nearest_distances = np.partition(distance_matrix, 1, axis=1)[:, 0]
    second_nearest_distances = np.partition(distance_matrix, 1, axis=1)[:, 1]
    edge_coefficients = nearest_distances / second_nearest_distances

    # STEP3: 区分边缘点和非边缘点
    non_edge_mask = edge_coefficients <= edge_threshold
    edge_mask = ~non_edge_mask

    non_edge_points = delivery_points[non_edge_mask]
    edge_points = delivery_points[edge_mask]

    # 初始化point_assignments
    point_assignments = np.full(delivery_points.shape[0], -1)

    # STEP4: 将非边缘点划入最近配送中心的服务范围
    nearest_centers = np.argmin(distance_matrix, axis=1)
    point_assignments[non_edge_mask] = nearest_centers[non_edge_mask]

    # STEP5: 分配边缘点给最近的客户点集,逐个更新
    # for i, edge_point in enumerate(edge_points):
    #     edge_point_index = np.where((delivery_points == edge_point).all(axis=1))[0][0]
    #     min_distance = float('inf')
    #     assigned_center = -1
    #     for center in range(distribution_centers.shape[0]):
    #         cluster_points = delivery_points[point_assignments == center]
    #         if cluster_points.size == 0:
    #             continue
    #         distances = cdist([edge_point], cluster_points).min()
    #         if distances < min_distance:
    #             min_distance = distances
    #             assigned_center = center
    #     point_assignments[edge_point_index] = assigned_center

    # STEP5: 分配边缘点给最近的客户点集，统一更新
    edge_assignments = np.full(edge_points.shape[0], -1)
    for i, edge_point in enumerate(edge_points):
        min_distance = float('inf')
        assigned_center = -1
        for center in range(len(distribution_centers)):
            cluster_points = delivery_points[point_assignments == center]
            if cluster_points.size == 0:
                continue
            distances = cdist([edge_point], cluster_points).min()
            if distances < min_distance:
                min_distance = distances
                assigned_center = center
        edge_assignments[i] = assigned_center
    # 最后统一更新point_assignments
    point_assignments[edge_mask] = edge_assignments

    return point_assignments
