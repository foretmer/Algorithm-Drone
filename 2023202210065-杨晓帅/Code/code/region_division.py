"""

@Author: So What
@Time: 2024/6/30 13:00
@File: region_division.py

"""
import numpy as np
import matplotlib.pyplot as plt

file_path_delivery_points_distances  = '../Data-points/delivery_points_distances.npy'
delivery_points_distances = np.load(file_path_delivery_points_distances, )

file_path_delivery_points  = '../Data-points/delivery_points.npy'
delivery_points = np.load(file_path_delivery_points, )

file_path_unloading_points  = '../Data-points/unloading_points.npy'
unloading_points = np.load(file_path_unloading_points, )

file_path_noise_points  = '../Data-points/noise_points.npy'
noise_points = np.load(file_path_noise_points,)

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

plt.title('Division Region Data Points', fontsize = 16)
plt.xlabel('X coordinate', fontsize = 14)
plt.ylabel('Y coordinate', fontsize = 14)
plt.legend()
plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
plt.axis('equal')  # 设置坐标轴比例以保持圆形为标准圆
plt.show()