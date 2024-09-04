"""

@Author: So What
@Time: 2024/6/29 20:03
@File: generate_points.py

"""
import numpy as np
import matplotlib.pyplot as plt
import json

def generate_points(num_delivery, num_unloading_per_delivery, num_noise, radius = 10, area_size = (100, 100)):
    """
    生成发货点和收货点，以及噪声点。
    """
    # 生成发货点
    delivery_points = np.random.rand(num_delivery, 2) * area_size

    # 生成卸货点，确保在对应的发货点的半径为10的圆内
    unloading_points = []
    for delivery_point in delivery_points:
        angles = np.random.rand(num_unloading_per_delivery) * 2 * np.pi
        radii = radius * np.sqrt(np.random.rand(num_unloading_per_delivery))
        unloadings = np.column_stack((delivery_point[0] + radii * np.cos(angles),
                                   delivery_point[1] + radii * np.sin(angles)))
        unloading_points.append(unloadings)

    unloading_points = np.vstack(unloading_points)

    # 生成噪声点
    noise_points = np.random.rand(num_noise, 2) * area_size

    return delivery_points, unloading_points, noise_points


def calculate_distances(points):
    """
    计算点集内所有点之间的距离矩阵。
    """
    return np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2).sum(axis = 2))

def circle_data(delivery_points, unloading_points, radius=10):
    """
    返回每个半径为10的圆内所有点的坐标和距离。
    """
    circle_info = []
    for i, center in enumerate(delivery_points):
        points_in_circle = [center]
        distances_to_center = np.sqrt(((unloading_points - center) ** 2).sum(axis=1))
        in_circle_points = unloading_points[distances_to_center <= radius]
        points_in_circle.extend(in_circle_points)
        distances = calculate_distances(np.array(points_in_circle))
        circle_info.append({
            'center': center,
            'points': np.array(points_in_circle),
            'distances': distances
        })
    return circle_info

# 设置发货点和收货点数量，以及噪声点数量
num_delivery = 5
num_unloading_per_delivery = 4
num_noise = 4

# 生成点
delivery_points, unloading_points, noise_points = generate_points(num_delivery, num_unloading_per_delivery, num_noise)

# 可视化
plt.figure(figsize = (12, 12))
ax = plt.gca()
for center in delivery_points:
    circle = plt.Circle(center, 10, color = 'blue', fill = False, linestyle = '--', linewidth = 2, alpha = 0.5)
    ax.add_artist(circle)
plt.scatter(delivery_points[:, 0], delivery_points[:, 1], color = 'dodgerblue', label = 'Delivery Points', s = 150,
            edgecolor = 'black', marker = 's')
plt.scatter(unloading_points[:, 0], unloading_points[:, 1], color = 'limegreen', label = 'Unloading Points', s = 100,
            edgecolor = 'black', alpha = 0.7)
plt.scatter(noise_points[:, 0], noise_points[:, 1], color = 'red', label = 'Noise Points', s = 50, edgecolor = 'black',
            marker = 'x')

plt.title('Visualization of Data Points', fontsize = 16)
plt.xlabel('X coordinate', fontsize = 14)
plt.ylabel('Y coordinate', fontsize = 14)
plt.legend()
plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
plt.axis('equal')  # 设置坐标轴比例以保持圆形为标准圆
plt.show()

# # 输出距离矩阵和圆内的收货点坐标
delivery_points_distances = calculate_distances(delivery_points)
print("Distance Matrix between Delivery Points:\n", calculate_distances(delivery_points))
# print("Distance Matrix between Unloading Points:\n", calculate_distances(unloading_points))
# for i, center in enumerate(delivery_points):
#     in_circle = np.sqrt(((unloading_points - center) ** 2).sum(axis = 1)) <= 10
#     print(f"Unloading points within 10 units of Delivery Point {i + 1}:\n", unloading_points[in_circle])

# 计算圆内的点间距离和坐标
circle_info = circle_data(delivery_points, unloading_points)
for info in circle_info:
    print(f"Center Point: {info['center']}")
    print("Points in Circle:")
    for point in info['points']:
        print(point)
    print("Distance Matrix within Circle:\n", info['distances'])
    print()




i = 1
for info in circle_info:
    file_path = 'class{}.npz'.format(i)
    np.savez(file_path, **info)
    i = i+1

file_path_delivery_points_distances  = 'delivery_points_distances.npy'
np.save(file_path_delivery_points_distances, delivery_points_distances)

file_path_delivery_points  = 'delivery_points.npy'
np.save(file_path_delivery_points, delivery_points)

file_path_unloading_points  = 'unloading_points.npy'
np.save(file_path_unloading_points, unloading_points)

file_path_noise_points  = 'noise_points.npy'
np.save(file_path_noise_points, noise_points)
