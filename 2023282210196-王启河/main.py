import numpy as np
from scipy.spatial.distance import cdist
import itertools
import matplotlib.pyplot as plt
""" 相关参数 """
# 配送中心数量
center_num = 3
# 卸货点数量
unloda_num = 8
# 无人机最大携带物品数量
max_take = 20
# 无人机最大飞行距离(km)
max_dis = 20
# 无人机飞行速度(km/h)
speed = 60
# 订单轮数
order_round = 15
# 设置随机种子以确保结果可重复
np.random.seed(15)
# order_num[i][j]表示第i轮订单，j卸货点的订单数量
order_num = np.random.randint(0, 8, size=(order_round, unloda_num))
# order_time[i][j]表示第i轮订单，j卸货点订单剩余时间,1为0.5小时，3为1.5小时，6为3小时
order_time = np.random.choice([1, 3, 6], size=(order_round, unloda_num), p=[0.8, 0.1, 0.1])
# # centers配送中心坐标
centers = np.array([[2.4, 1.8], [12.1, 2.5], [8.1, 12.2]])
# # unlodas卸货点坐标
unlodas = np.array([[1.3, 6.1],
                    [5.2, 6.1],
                    [4.2, 9.7],
                    [7.1, 3.8],
                    [10.2, 6.5],
                    [11.5, 9.5],
                    [4.5, 12.5],
                    [8.5, 8.4]])

""" 算法函数 """
def distance(p1, p2):
    """ 计算两点之间的欧几里得距离 """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def find_cycles(centers, unloads, max_length, max_take):
    """ 找到所有以 'centers' 点为起点和终点的回路，
        经过至少一个给定坐标点，
        长度小于 'max_length'且节点数小于max_take的所有回路
        输入：centers 配送中心
             unloads 卸货点
             max_length 无人机最长飞行距离
             max_take 无人机最大载重
        输出：
             cycles合法回路"""
    n = len(unloads)
    cycles = set()  # 使用集合存储唯一的回路及其长度
    seen_paths = set()  # 记录已经添加过的路径及其逆序

    # 遍历所有起点
    for s in range(len(centers)):
        # 遍历所有坐标的排列，考虑每个可能的回路
        for perm in itertools.permutations(list(range(0, n))):
            # 从起点开始构建回路，经过排列中的每个坐标点，再回到起点
            for i in range(n):
                path = list(perm[:i + 1])
                size = len(path)
                length = 0
                # 计算回路的总长度
                for j in range(size - 1):
                    length += distance(unloads[path[j]], unloads[path[j + 1]])
                length += distance(unloads[path[0]], centers[s]) + distance(unloads[path[size - 1]], centers[s])
                # 检查回路是否有效且长度小于 max_length
                if length < max_length and size <= max_take:
                    # 将路径和其逆序分别转换为元组
                    path = [s] + path + [s]
                    forward_path = tuple(path)
                    backward_path = tuple(path[::-1])
                    # 如果路径和其逆序都没有出现过，则将路径及其逆序添加到集合中
                    if forward_path not in seen_paths and backward_path not in seen_paths:
                        cycles.add((s, forward_path[1:-1], length))
                        seen_paths.add(forward_path)
                        seen_paths.add(backward_path)
    return cycles


def cycles_sort(order, cycles, max_take):
    """ 计算cycles权重，并排序返回"""
    weighted_cycles = []
    for cycle in cycles:
        unloads_weight = 0
        unlodas = cycle[1]
        tmp_order = order.copy()
        # 获得无人机经过回路分配后，归零卸货点订单数量z
        act, tmp_order, z = allocation(tmp_order, unlodas, max_take)
        for unload in unlodas:
            unloads_weight += order[unload]
        weight = cycle[2]
        # 计算权重
        if unloads_weight <= max_take:
            weight /= (unloads_weight * (z + 1))
        else:
            weight /= (max_take * (z + 1))
        # 添加权重信息到元组中
        weighted_cycles.append((cycle[0], cycle[1], cycle[2], weight))
    # 排序
    sorted_cycles = sorted(weighted_cycles, key=lambda x: x[3])
    return sorted_cycles


def delete_cycle(order, cycles):
    """ 删除订单数量为0的卸货点所在的所有回路"""
    cycles_to_remove = set()  # 使用集合来收集需要删除的回路
    for i in range(len(order)):
        if order[i] == 0:
            for cycle in cycles:
                if i in cycle[1]:
                    cycles_to_remove.add(cycle)
    # 删除回路
    for cycle in cycles_to_remove:
        cycles.remove(cycle)
    return cycles


def allocation(order, cycle_unloads, max_take):
    """ 一条回路上一台无人机载重分配
        输入：
            order[i]为i卸货点需要派送的订单数量
            cycle_unloads是回路上的卸货点
            max_take为无人机最大载重
        输出：
            act[j]: 无人机第j个卸货点卸货数量
            order: 剩余订单数量
            zero_num: 分配后订单清零节点数量"""
    n = len(order)
    act = [0] * n
    # 如果一台无人机可以送完回路所有卸货点订单，无人机分配就按订单数量分配
    for unload in cycle_unloads:
        act[unload] = order[unload]
    # 如果一台无人机无法送完回路所有卸货点订单，就尽量均匀地分配到所有订货点
    if sum(act) > max_take:
        i = 0
        act = [0] * n
        while max_take:
            unload = cycle_unloads[i]
            if order[unload] > act[unload]:
                act[unload] += 1
                max_take -= 1
            i = (i + 1) % len(cycle_unloads)
    zero_num = 0
    for i in range(n):
        if order[i] == act[i] and order[i] != 0:
            zero_num += 1
        order[i] -= act[i]
    return act, order, zero_num


def delivery(order, cycles, max_take):
    """ 单时刻需发送订单的无人机派送算法
        输入：
            oreder[i]为i卸货点需要派送的订单数量
            cycles是所有回路
            max_take为无人机最大载重
        输出：
            deliver_messages无人机派送回路相关信息：分配中心编号，回路路径，回路距离, 物品分配"""
    tmp_cycles = cycles.copy()
    deliver_messages = []
    # 卸货点还有订单，则还需派出无人机
    while sum(order) != 0:
        # 删除存在无订单卸货点的回路
        tmp_cycles = delete_cycle(order, tmp_cycles)
        # 根据剩余订单对回路排序
        tmp_cycles = cycles_sort(order, tmp_cycles, max_take)
        best_cycle = tmp_cycles[0]
        # 根据最优回路和剩余订单对当前无人机载货进行分配
        center = best_cycle[0]
        path = best_cycle[1]
        path_dis = best_cycle[2]
        act, order, z = allocation(order, best_cycle[1], max_take)
        # 添加无人机配送方案
        deliver_messages.append((center, path, path_dis, act))
        unload_plan = f'center{center} -> '
        for p in path:
            unload_plan += f'(unload{p},{act[p]}) -> '
        unload_plan += f'center{center}'
        print(f'--------------center{center}派出一架无人机------------------')
        print(f'无人机卸货方案：\n{unload_plan}')
        print(f'配送距离：{path_dis}')
    return deliver_messages



def plot_path(centers, unloads, deliver_messages, order, time):
    """ 绘制派送路径 """

    # 格式化起点的坐标并保留一位小数
    center_x = [round(point[0], 1) for point in centers]
    center_y = [round(point[1], 1) for point in centers]

    # 格式化给定坐标点的 x 和 y 坐标并保留一位小数
    unloads_x = [round(point[0], 1) for point in unloads]
    unloads_y = [round(point[1], 1) for point in unloads]

    # 创建新的图形
    plt.figure(figsize=(8, 6))

    # 设置 x 轴和 y 轴的范围从 0 开始
    plt.xlim(0, 18)  # x 轴从 0 开始，上限自动设置
    plt.ylim(0, 18)  # y 轴从 0 开始，上限自动设置

    # 绘制起点
    plt.scatter(center_x, center_y, color='red', label='centers')
    # 绘制给定的坐标点
    plt.scatter(unloads_x, unloads_y, color='blue', label='unloads')

    colors = plt.cm.rainbow  # 使用彩虹色系列
    # 绘制回路
    idx = 0
    total_distance = 0
    for deliver_message in deliver_messages:
        center_idx = deliver_message[0]
        unloads_indices = deliver_message[1]
        total_distance += deliver_message[2]
        act = deliver_message[3]
        # 获取配送中心和回路上卸货点坐标
        center_crood = centers[center_idx]
        unloads_crood = [unloads[i] for i in unloads_indices]

        color = colors(idx / len(deliver_messages))  # 根据回路的索引选择颜色

        # 无人机物品投放数量
        act_str = '(' + ','.join(map(str, act)) + ')'

        # 绘制起点到第一个节点的连线
        plt.plot([center_crood[0], unloads_crood[0][0]], [center_crood[1], unloads_crood[0][1]], color=color, label=f'UAV{idx+1}:{act_str}')

        # 绘制节点之间的连线
        for k in range(len(unloads_crood) - 1):
            plt.plot([unloads_crood[k][0], unloads_crood[k + 1][0]], [unloads_crood[k][1], unloads_crood[k + 1][1]], color=color)

        # 绘制最后一个节点到起点的连线
        plt.plot([unloads_crood[-1][0], center_crood[0]], [unloads_crood[-1][1], center_crood[1]], color=color)
        idx += 1

    # 添加坐标点的标签，并格式化为一位小数
    for i, txt in enumerate(centers):
        plt.annotate(f'C{i}', (center_x[i], center_y[i]), textcoords="offset points", xytext=(5,5), ha='left')
    for i, txt in enumerate(unloads):
        plt.annotate(f'U{i}: {order[i]}', (unloads_x[i], unloads_y[i]), textcoords="offset points", xytext=(5,5), ha='left')

    # 添加图例和标题
    plt.legend()
    plt.title(f'Round:{time}  Total Distance: {round(total_distance, 2)} km')
    plt.xlabel('X(km)')
    plt.ylabel('Y(km)')

    # 保存图形
    plt.savefig(f'./result/Round_{time}.png')
    # 显示图形
    plt.grid(True)
    plt.show()




def GetOrders(order_num, order_time):
    """ 根据订单数量和订单时间生成每轮必须发的订单
        若只剩0.5小时，则必须发
        输入：order_num 订单数量
             order_time 订单剩余时间（0.5, 1.5. 3)对应（1， 3， 6）
        输出：orders是必发订单"""
    m = len(order_num)
    n = len(order_num[1])
    # 初始化一个必发订单
    orders = [[0 for i in range(n)] for i in range(m)]
    # 计算必发订单
    for i in range(m):
        for j in range(n):
            # 计算当前订单最晚发的时间
            must_round = min(i + order_time[i][j], m) - 1
            orders[must_round][j] += order_num[i][j]
    return orders


# 生成必发订单
orders = GetOrders(order_num, order_time)
# 随机生成必发订单
# orders = np.random.randint(0, 10, size=(order_round, unloda_num))
# 找到所有环
cycles = find_cycles(centers, unlodas, max_dis, max_take)
# 对每时刻订单进行处理
time = 0
all_distance = 0
all_AUV_num = 0
for order in orders:
    time += 1
    print(f'##############第{time}轮无人机配送情况###############')
    order_tmp = order.copy()
    print(f'卸货点订单数量：{order}')
    deliver_messages = delivery(order_tmp, cycles, max_take)
    distance = sum(deliver_message[2] for deliver_message in deliver_messages)
    AUV_num = len(deliver_messages)
    all_distance += distance
    all_AUV_num += AUV_num
    print('.................................................')
    print(f'第{time}轮共派出{AUV_num}架无人机, 配送总长为{round(distance, 2)} km')
    plot_path(centers, unlodas, deliver_messages, order, time)

print('**************全局情况********************')
print(f'一共派出{all_AUV_num}架无人机，派送总距离为：{round(all_distance, 2)} km')






