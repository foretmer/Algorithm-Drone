import numpy as np
import matplotlib.pyplot as plt


def display_all_path_with_labels(distribution_centers, delivery_points, paths, map_size, active_delivery_points_indices):
    plt.figure(figsize=(15, 15))
    distribution_centers_np = np.array(distribution_centers)
    delivery_points_np = np.array(delivery_points)
    plt.scatter(distribution_centers_np[:, 0], distribution_centers_np[:, 1], c='red', label='Distribution Centers', s=20)

    plt.scatter(delivery_points_np[:, 0], delivery_points_np[:, 1], c='red', label='Delivery Points', s=20)
    # # 仅绘制有订单的卸货点
    # active_delivery_points_np = np.array([delivery_points[i] for i in active_delivery_points_indices])
    # plt.scatter(active_delivery_points_np[:, 0], active_delivery_points_np[:, 1], c='blue', label='Active Delivery Points', s=20)

    colors = ['g', 'b', 'm', 'c', 'y', 'k']  # 使用不同颜色区分不同配送中心的路径
    for i, (center, planned_path) in enumerate(paths):
        color = colors[i % len(colors)]
        plt.plot(planned_path[:, 0], planned_path[:, 1], color + '-', linewidth=1)

    # 标记每个配送中心和有订单的卸货点的编号
    for idx, point in enumerate(distribution_centers_np):
        plt.text(point[0], point[1], f'DC{idx}', fontsize=12, ha='right')
    for idx, point in enumerate(delivery_points):
        plt.text(point[0], point[1], str(idx), fontsize=12, ha='right')
    # # 仅绘制有订单的卸货点
    # for idx, point in zip(active_delivery_points_indices, active_delivery_points_np):
    #     plt.text(point[0], point[1], str(idx), fontsize=12, ha='right')

    plt.title('Delivery Routes for All Orders')
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, map_size)
    plt.ylim(0, map_size)
    plt.show()


# 可视化聚合结果
def display_aggregate(distribution_centers, delivery_points, full_point_assignments, map_size):
    plt.figure(figsize=(15, 15))
    distribution_centers_np = np.array(distribution_centers)
    delivery_points_np = np.array(delivery_points)
    plt.scatter(distribution_centers_np[:, 0], distribution_centers_np[:, 1], c='red', label='Distribution Centers', s=20)
    plt.scatter(delivery_points_np[:, 0], delivery_points_np[:, 1], c='blue', label='Delivery Points', s=20)

    # 标注每个点的分配情况
    for i, point in enumerate(delivery_points_np):
        if full_point_assignments[i] != -1:
            plt.text(point[0], point[1], str(full_point_assignments[i]), fontsize=12, ha='right')

    plt.title('Aggregation Optimization of Delivery Points')
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, map_size)
    plt.ylim(0, map_size)
    plt.show()