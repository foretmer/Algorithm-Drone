"""

@Author: So What
@Time: 2024/6/30 18:09
@File: overlapping_analysis.py

"""
"""
返回结果如下，从结果可以看出配送中心$j_2,j_5$之间的距离小于20。
因此，我们计算位于$j_2$和$j_5$对应的两个责任圆中所有点之间的距离，返回结果如下。
"""
import numpy as np
import matplotlib.pyplot as plt

def calculate_distances(points):
    """
    计算点集内所有点之间的距离矩阵。
    """
    return np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2).sum(axis = 2))

def find_duplicate_rows(array1, array2):
    """
    Find and return the duplicate rows between two NumPy arrays.
    """
    common_rows = np.intersect1d(array1.view([('', array1.dtype)] * array1.shape[1]),
                                 array2.view([('', array2.dtype)] * array2.shape[1]))
    # Convert the structured array back to a regular 2D array
    common_rows = common_rows.view(array1.dtype).reshape(-1, array1.shape[1])
    return common_rows

def plot_2d_array_with_highlight(array1, array2):
    """
    Parameters:
    array1 (np.ndarray): Input 2D NumPy array where each row represents a point (x, y).
    array2 (np.ndarray): 2D NumPy array with points to be highlighted.
    """
    if array1.shape[1] != 2 or array2.shape[1] != 2:
        raise ValueError("Both input arrays must be 2D arrays with shape (n, 2)")

    # Extract x and y coordinates for array1
    x1 = array1[:, 0]
    y1 = array1[:, 1]
    plt.figure(figsize = (8, 6))
    ax = plt.gca()
    # Extract x and y coordinates for array2
    x2 = array2[:, 0]
    y2 = array2[:, 1]
    for center in array2:
        circle = plt.Circle(center, 10, color = 'blue', fill = False, linestyle = '--', linewidth = 2, alpha = 0.5)
        ax.add_artist(circle)
    # Create the plot
    plt.scatter(x1, y1, color = 'limegreen', label = 'Unloading Points', s = 100,
                edgecolor = 'black', alpha = 0.7 )
    plt.scatter(x2, y2, color = 'dodgerblue', label = 'Delivery Points', s = 150,
                edgecolor = 'black', marker = 's')
    # Annotate all points in array1
    for i in range(len(x1)):
        plt.annotate(f'({x1[i]:.2f}, {y1[i]:.2f})', (x1[i], y1[i]),
                     textcoords = "offset points", xytext = (0, 10),
                     ha = 'center', fontsize = 8, color = 'black')

    # Annotate all points in array2
    for i in range(len(x2)):
        plt.annotate(f'({x2[i]:.2f}, {y2[i]:.2f})', (x2[i], y2[i]),
                     textcoords = "offset points", xytext = (0, 10),
                     ha = 'center', fontsize = 8, color = 'black')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.legend()
    plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
    plt.axis('equal')  # 设置坐标轴比例以保持圆形为标准圆
    plt.show()

if __name__ == '__main__':
    # 拿到有重合责任圆的两个配送中心及其卸货点的数据点
    file_path_j2 = '../Data-points/class2.npz'
    file_path_j5 = '../Data-points/class5.npz'
    j2_data_points = np.load(file_path_j2)
    j5_data_points = np.load(file_path_j5)
    j2_and_j5_data_points = np.concatenate((j2_data_points['points'],j5_data_points['points']))
    j2_and_j5_data_points = np.unique(j2_and_j5_data_points,axis = 0)#去掉重合卸货点
    overlapping_points = find_duplicate_rows(j2_data_points['points'],j5_data_points['points'])#得到重合的卸货点
    print('overlapping_points: \n',overlapping_points)
    center_points = np.array([[40.78822419,12.58522706],[33.72449907,23.95258972]])
    plot_2d_array_with_highlight(j2_and_j5_data_points,center_points)
