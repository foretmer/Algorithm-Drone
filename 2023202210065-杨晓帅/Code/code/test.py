"""

@Author: So What
@Time: 2024/6/29 20:16
@File: test.py

"""
import numpy as np
import matplotlib.pyplot as plt
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
    plt.figure(figsize = (8, 8))
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

center_points = np.array([[41.00483389,62.06192927]])
unloading_points = np.array( [[33.83745997, 67.1227674],
                              [40.3364198, 57.55052412],
                              [48.90133248, 64.15187898],
                              [38.07218189, 53.09899463]])
plot_2d_array_with_highlight(unloading_points,center_points)