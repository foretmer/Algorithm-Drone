import matplotlib.pyplot as plt


def is_urgent(x, now):
    return x - now >= 10


def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def dis_matrix(position1, position2):
    distance_matrix = []
    for i_xy in position1:
        dis_ls = []
        for j_xy in position2:
            dis_tmp = calculate_distance(i_xy, j_xy)
            # 1、距离配货点超过10不可达，设置为100
            # 2、卸货点间超过10没必要派无人机遍历、双无人机的路径最差结果比单无人机好，且运送订单是单无人机的2倍
            dis_tmp = round(dis_tmp, 2) if dis_tmp <= 10 else 100
            dis_ls.append(dis_tmp)
        distance_matrix.append(dis_ls)
    return distance_matrix


def draw_lines(points1, points2, distance_dp, distance_dy):
    x, y = [], []
    for xy in points1:
        x.append(xy[0])
        y.append(xy[1])
    plt.scatter(x, y, color='red', marker='*', s=100)  # 使用scatter函数绘制散点图
    x, y = [], []
    for xy in points2:
        x.append(xy[0])
        y.append(xy[1])
    plt.scatter(x, y, color='blue', s=50)  # 使用scatter函数绘制散点图

    for i in range(len(distance_dp)):
        for j in range(len(distance_dp) - i - 1):
            if distance_dp[i][j + i + 1] <= 10:
                plt.plot([points2[i][0], points2[j + i + 1][0]],
                         [points2[i][1], points2[j + i + 1][1]], color='green')
    for i in range(len(distance_dy)):
        for j in range(len(distance_dy[i])):
            if distance_dy[i][j] <= 10:
                plt.plot([points1[i][0], points2[j][0]],
                         [points1[i][1], points2[j][1]], color='green')

    # 设置图形属性
    plt.title('Address Scatter Plot with Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # 显示图形
    plt.grid(True)
    plt.savefig("AddressLines.png")
    plt.show()


def find_cycles_sat(graph_to_find):
    visited = set()
    cycles = []
    cycles_len = []

    def dfs(node_now, start, path, weights):
        visited.add(node_now)
        for neighbor, weight in graph_to_find.get(node_now, {}).items():
            if neighbor == start and len(path) > 1:
                cycles.append(path + [neighbor])
                cycles_len.append(weights + [round(weights[-1] + weight, 2)])
            elif neighbor not in visited:
                dfs(neighbor, start, path + [neighbor], weights + [round(weights[-1] + weight, 2)])
        visited.remove(node_now)

    for node in graph_to_find:
        if 'dy' in node:
            dfs(node, node, [node], [0.0])

    cycles_sat = []
    cycles_len_sat = []
    for routes, route_lens in zip(cycles, cycles_len):
        if route_lens[-1] <= 20:
            flag = True
            for cycle in cycles_sat:
                reversed_cycle = cycle[::-1]
                if reversed_cycle == routes:  # 如果两个回路相等，则不加入路径list
                    flag = False
                    break
            if flag is True:
                cycles_sat.append(routes)
                cycles_len_sat.append(route_lens)
                print(routes, route_lens, route_lens[-1])
    print(cycles_sat, cycles_len_sat)
    return cycles_sat, cycles_len_sat


def duplicate_removal(A, B):
    if A is None or B is None:
        return []
    return list(set(A).intersection(set(B)))
