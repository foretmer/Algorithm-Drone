"""

@Author: So What
@Time: 2024/6/30 13:11
@File: data_analysis.py

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
for i in range(5):
    file_path = '../Data-points/class{}.npz'.format(i+1)
    data = np.load(file_path)
    # print(data['center'])
    # print((data['points']))
    print((data['distances']))
    # print((data['distances'])/60*60)


# file_path_delivery_points_distances  = '../Data-points/delivery_points_distances.npy'
# delivery_points_distances = np.load(file_path_delivery_points_distances, )
# print(delivery_points_distances)
# print(np.ceil(delivery_points_distances,))





