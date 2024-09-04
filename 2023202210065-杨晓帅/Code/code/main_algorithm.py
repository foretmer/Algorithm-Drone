"""

@Author: So What
@Time: 2024/6/30 13:03
@File: main_algorithm.py

"""
import numpy as np
import sys
# print(data['center'])
# print((data['points']))
# print((data['distances']))

N = 3 #节点数
D = np.zeros((N, N)).astype(int)
S = set()  #对应点的集合
for i in range(1, N):
    S.add(i)
# file_path = '../Data-points/class{}.npz'.format(1)
# data = np.load(file_path)
# print(data['center'])
# print('----------')
# print(data['points'])
# D = data['distances'][0:]
# print('----------')
# print(D)
D  = np.array([[ 0. ,          8.77401456,   4.56065279 ,  8.16838906 ],
                [ 8.77401456,  0.  ,         11.56997495,  15.35403638],
                [ 4.56065279,  11.56997495,  0.,           10.81367723],
                [ 8.16838906,  15.35403638,  10.81367723 , 0.         ]])

D  = np.array([[ 0. ,          8.77401456,   4.56065279 ,],
                [ 8.77401456,  0.  ,         11.56997495,],
                [ 4.56065279,  11.56997495,  0.,         ],])

print(D)
def  get_subset(point_set, n): #根据点数构造子集
    if n == 0:
        return []
    sub_sets = []
    for p in point_set:
        s = set()
        s.add(p)
        set_copy = point_set.copy()
        set_copy.discard(p)
        sets = get_subset(set_copy, n - 1)
        if len(sets) == 0:
            sub_sets.append(s)
        else:
            for each_set in sets:
                s_union = s.copy().union(each_set)
                sub_sets.append(s_union)
    return sub_sets


def  find_shortest_path():  #计算最短路径
    C = {}
    for point_count in range(1, N):
        sub_sets = get_subset(S.copy(), point_count)  #获得包含给定点数的子集
        for the_set in sub_sets:
            distances = {}  #记录起始点到集合内每一点的最短距离
            for the_point in the_set: #计算当前集合内的点作为回到起始点前一个点时对应的最短距离
                after_discard = the_set.copy()
                after_discard.discard(the_point)
                if len(after_discard) == 0:
                    distances[the_point] = D[0][the_point]
                else:
                    '''
                    如果集合S包含三个点{1,2,3},从节点0开始遍历集合中所有点的最短路径算法为,先找出从起始点0开始遍历集合{1,2},{1,3},{2,3}的最短路径，
                    然后用这三条路径长度分别加上D[2][3],D[3][2],D[3][1]，三个结果中的最小值就是从起始点0开始，遍历集合{1,2,3}中所有点后的最短路径.
                    因此当集合为{1,2,3}时，C[{1,2,3}]对应key值为1，2，3的一个map对象，map[1]表示集合点为{1,2,3}时，遍历路径最后一个节点时1时的最短距离
                    '''
                    set_copy = the_set.copy()
                    set_copy.discard(the_point)
                    distance_to__points = C[frozenset(set_copy)]
                    d = sys.maxsize
                    for p in set_copy:
                        if D[p][the_point] + distance_to__points[p] < d:
                            d = D[p][the_point] + distance_to__points[p]
                    distances[the_point] = d
            C[frozenset(the_set)] = distances.copy() #记录起始点到当前集合内每个点的最短距离,根据python语法,fronzenset才能作为key
            distances.clear()
    distances = C[frozenset(S)]
    d = sys.maxsize
    for point in S: #先找到起始点到集合{1,2...n}中对应点的嘴短距离
        if distances[point] + D[point][0] < d:
            d = distances[point] + D[point][0]
    return d, C
shortest_distance, C = find_shortest_path()
print("the shortest distance for visiting all points is : ", shortest_distance)

def  find_visiting_path(point_set, C):
    if len(point_set)== 1:
        return [point_set.pop()]
    distances = C[frozenset(point_set)]
    d = sys.maxsize
    path_point = []
    selected_point = None
    for point in S: #先找到起始点到集合{1,2...n}中对应点的嘴短距离
        if distances[point] + D[point][0] < d:
            d = distances[point] + D[point][0]
            selected_point = point
    point_set.discard(selected_point)
    points_before = find_visiting_path(point_set, C)
    path_point.extend(points_before)
    path_point.extend([selected_point])
    return path_point
points =    find_visiting_path(S, C)
print("the shortest path for visiting all points is : ", points)