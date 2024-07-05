import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import art3d
import numpy as np


def draw_graph(centers, points):
    # 覆盖半径
    radius = 10

    # 绘制图形
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 20})

    # 绘制配送中心和覆盖圆
    for i, center in enumerate(centers):
        plt.scatter(center[0], center[1], c='red', marker='s', label=f'Center {i + 1}')
        circle = plt.Circle(center, radius, color='red', fill=False, linestyle='--')
        plt.gca().add_patch(circle)

    # 绘制卸货点
    for i, drop in enumerate(points):
        plt.scatter(drop[0], drop[1], c='blue', marker='o')

    # 设置图例和标题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend(loc='upper right')
    plt.title('配送中心和卸货点坐标图')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)
    plt.xlim()
    plt.ylim()
    plt.gca().set_aspect('equal', adjustable='box')  # 保持圆形比例
    plt.show()


def draw_graph_with_route(centers, points, best_route, current_time):
    # 覆盖半径
    radius = 10

    # 绘制图形
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 20})

    # 绘制配送中心和覆盖圆
    for i, center in enumerate(centers):
        plt.scatter(center[0], center[1], c='red', marker='s', label=f'Center {i + 1}')
        circle = plt.Circle(center, radius, color='red', fill=False, linestyle='--')
        plt.gca().add_patch(circle)

    # 绘制卸货点
    for i, drop in enumerate(points):
        plt.scatter(drop[0], drop[1], c='blue', marker='o')

    # 绘制最佳路径
    for i in range(len(best_route)):
        for route in best_route[i]:
            if route:
                route_points = [centers[i]] + [order[1] for order in route] + [centers[i]]
                x, y = zip(*route_points)
                plt.plot(x, y, linewidth=2, alpha=0.5)

    # 设置图例和标题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend(loc='upper right')
    plt.title(f'T={current_time}时，无人机配送路线图')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)
    plt.xlim()
    plt.ylim()
    plt.gca().set_aspect('equal', adjustable='box')  # 保持圆形比例
    plt.show()

def draw_3d_graph_with_route(center, points, best_route, current_time):
    # 覆盖半径
    radius = 10

    # 创建3D图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams.update({'font.size': 20})

    # 绘制配送中心和覆盖圆
    ax.scatter(center[0], center[1], 0, c='red', marker='s', label='Center')
    circle = plt.Circle((center[0], center[1]), radius, color='red', fill=False, linestyle='--')
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=0, zdir="z")

    # 绘制卸货点
    for i, drop in enumerate(points):
        ax.scatter(drop[0], drop[1], 0, c='blue', marker='o')
    # 绘制最佳路径
    for i, route in enumerate(best_route):
        if route:
            route_points = [center] + [order[1] for order in route] + [center]
            x, y = zip(*route_points)
            z = np.linspace(0, 10, len(route_points))  # 生成高度变化，模拟飞行路径
            ax.plot(x, y, z, linewidth=2, alpha=0.7, label=f'Route {i+1}')

    # 设置图例和标题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax.legend(loc='upper right')
    ax.set_title(f'T={current_time}时，无人机配送路线图')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_zlabel('高度')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_zlim(0, 15)
    plt.show()

def draw_Fitness_Over_Generations(best_fitnesses, current_time, center):
    # 绘制适应度函数下降曲线
    plt.rcParams.update({'font.size': 10})
    plt.plot(best_fitnesses)
    plt.xlabel('迭代轮数(Generations)')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Improvement Over Generations of center{} at T={}'.format(center, current_time))
    plt.show()