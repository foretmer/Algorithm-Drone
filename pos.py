import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

distribution_centers = [(0, 0), (5, 5), (-5, 5)]    # 配送中心坐标
unloading_points = [(2, -3), (-4, 6), (3, 4), (-5, -2), (-3, 0)]    # 卸货点坐标
dc_x, dc_y = zip(*distribution_centers)     # x和y轴分离开
up_x, up_y = zip(*unloading_points)

# 1 计算各点的距离
distances = []
# 计算配送中心与卸货点之间的距离
print('******* Distances *******')
all_points = [(0, 0), (5, 5), (-5, 5), (2, -3), (-4, 6), (3, 4), (-5, -2), (-3, 0)]
for i in all_points:
    for j in all_points:
        distance = math.sqrt((i[0] - j[0])**2 + (i[1] - j[1])**2)
        print(f'i {i}, j {j}: {distance:.2f}')

# 2 画图
plt.figure(figsize=(10, 10))
# 配送中心
plt.scatter(dc_x, dc_y, color='red', label='配送中心', s=100)
for dc in distribution_centers:
    plt.text(dc[0], dc[1], f'{dc}', fontsize=12)

# 卸货点
plt.scatter(up_x, up_y, color='blue', label='卸货点', s=100)
for up in unloading_points:
    plt.text(up[0], up[1], f'{up}', fontsize=12)

plt.legend()
plt.title('配送中心和卸货点分布图')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
