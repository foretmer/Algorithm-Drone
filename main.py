import random
import math

max_dist = 20       # 无人机最大飞行距离
t = 10                  # 每隔t分钟生成一次订单并做决策
total_time = 24 * 60    # 一天的总时间（分钟）
m = 5                   # 每次生成0-m个订单
# 固定的配送中心和卸货点
distribution_centers = [(0, 0), (5, 5), (-5, 5)]
unloading_points = [(2, -3), (-4, 6), (3, 4), (-5, -2), (-3, 0)]

class Order:
    def __init__(self, id, x, y, remain_time, priority):
        self.id = id
        self.x = x
        self.y = y
        self.priority = priority
        self.remain_time = remain_time
        self.flag = False

    def __lt__(self, other):
        return self.remain_time > other.remain_time     # 剩余时间少的在前

class Drone:
    def __init__(self, home_distribution_center):
        self.capacity = 3                   # 无人机一次最多携带的物品数量
        self.max_distance = max_dist        # 最大飞行距离(公里)
        self.speed = 60                     # 速度(公里/小时)
        self.home_distribution_center = home_distribution_center    # 所属配送中心
        self.path = []              # 配送路径 旅行商回路
        self.current_load = 0       # 当前负载
        self.current_distance = 0   # 当前距离

    def can_carry(self):
        return self.current_load < self.capacity

    def prim_algorithm(self, pth):
        all_points = list(set(pth))     # 去除重复元素, 但注意load得+1
        # 从配送中心开始应用prim算法
        start = self.home_distribution_center
        visited = {start}
        unvisited = set(all_points)
        path = []
        while len(visited) <= len(all_points):  # visited比all_points多个配送中心 所以有=
            # 找visited和unvisited的距离最小的点
            min_dist = 99999
            for point in visited:
                for un_point in unvisited:
                    dist = distance(un_point, point)
                    if dist < min_dist:
                        min_dist = dist
                        visit_point = un_point  # 该访问这个节点了
            path.append(visit_point)
            visited.add(visit_point)
            unvisited.remove(visit_point)
        return path

    def path_and_dist(self, pth):
        pth = self.prim_algorithm(pth)  # TSP
        dist = sum(distance(pth[i], pth[i + 1]) for i in range(len(pth) - 1))
        dist = dist + distance(pth[0], self.home_distribution_center) + distance(pth[len(pth) - 1], self.home_distribution_center)    # 回到原点
        return pth, dist

    def test_order(self, order):    # 这里只用来计算路径和长度, 没有真的更新无人机(update_order)
        if self.can_carry():
            pth = self.path + [(order.x, order.y)]   # 直接添加这个点
            # 计算当前路径和距离
            pth, dist = self.path_and_dist(pth)     # 计算最短路径, 先不更新self.path和self.distance
            if dist <= max_dist:
                return pth, dist
            else:   # 距离太大
                return [], 0
        return [], 0    # 装不下了

    def update_order(self, pth, dist):      # 确定是最小才更新无人机的运送情况
        self.current_load += 1
        self.path = pth
        self.current_distance = dist

# 计算两点之间的距离
def distance(p1, p2):
    distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return distance

# 随机生成0-m个订单
def generate_orders(current_time):
    orders = []
    print(f"时间 {current_time}: 生成的订单信息如下:")
    for up in unloading_points:
        num_orders = random.randint(0, m)   # 为每个卸货点生成0-m个订单
        x, y = up                           # 卸货点的坐标
        if num_orders != 0:
            print(f"卸货点: {up}")
        for i in range(num_orders):
            remain_time = random.choice([30, 90, 180])  # 随机生成该订单的紧急程度
            if remain_time == 30:
                priority = '紧急'
            elif remain_time == 90:
                priority = '较紧急'
            elif remain_time == 180:
                priority = '一般'
            order = Order(i, x, y, remain_time, priority)
            orders.append(order)
            print(f"订单ID: {order.id}, 优先级: {order.priority}, 剩余时间: {order.remain_time}")
    return orders

def greedy_delivery(orders, current_time):
    total_distance = 0
    drones = []
    # 按剩余时间对订单进行排序
    orders.sort(reverse=True)

    # 剩余时间不足t+20=30分钟(下一趟还有10min, 一趟来回最远20km, 即20min; 下一趟送的话就需要等10min后配送20min)的必须在本轮送出去,
    # 剩下的如果无人机还有空位且最短路径不会超过20km可以此轮送走
    # 剩余时间不足30分钟: 如果还没有无人机, 找一个离订单最近的配送中心, 新建一架无人机;
    # 如果有无人机, 对比是加入已有无人机中距离最短, 还是找一个离订单最近的配送中心, 新建一架无人机往返的距离最短
    # flag = true

    # # 剩下的订单: 尽量保证无人机是装满的, 且满足20km的前提: 如果有无人机, 可以考虑是否把订单加入到这几个无人机中; 没有就算了
    # # 不装走的话, remain_time - t (间隔时间)

    for order in orders:
        if order.remain_time <= 30:
            # 如果还没有无人机, 找一个离订单最近的配送中心, 新建一架无人机;
            if len(drones) == 0:
                nearest_dc = min(distribution_centers, key=lambda dc: distance(dc, (order.x, order.y)))
                new_drone = Drone(nearest_dc)  # 从最近的配送中心生成新的无人机
                min_dist = 2 * distance(nearest_dc, (order.x, order.y))
                new_drone.update_order([(order.x, order.y)], min_dist)
                drones.append(new_drone)
            # 如果有无人机, 对比是加入已有无人机中距离最短, 还是找一个离订单最近的配送中心, 新建一架无人机往返的距离最短
            else:
                min_dist, ind = 999999, -1
                best_pth = []
                for i, drone in enumerate(drones):
                    pth, dist = drone.test_order(order)
                    if dist != 0:
                        if dist < min_dist:
                            min_dist = dist
                            ind = i
                            best_pth = pth
                nearest_dc = min(distribution_centers, key=lambda dc: distance(dc, (order.x, order.y)))
                new_drone = Drone(nearest_dc)
                dist = 2 * distance(nearest_dc, (order.x, order.y))
                # new_drone.update_order([nearest_dc], dist)
                if dist < min_dist:
                    min_dist = dist
                    new_drone.update_order([(order.x, order.y)], dist)
                    drones.append(new_drone)
                else:
                    drones[ind].update_order(best_pth, min_dist)
            order.flag = True

        # 剩下的订单: 尽量保证无人机是装满的, 且满足20km的前提: 如果有无人机, 可以考虑是否把订单加入到这几个无人机中;
        # 没有就算了
        # 不装走的话, remain_time - t (间隔时间)
        else:
            min_dist, ind = 999999, -1
            best_pth = []
            for i, drone in enumerate(drones):
                pth, dist = drone.test_order(order)
                if dist != 0:
                    if dist < min_dist:
                        min_dist = dist
                        ind = i
                        best_pth = pth
            if ind != -1:   # 如果找到了可以装这个物品的无人机
                drones[ind].update_order(best_pth, min_dist)
                order.flag = True
            else:
                order.remain_time -= t

    # 最后计算所有无人机的dist
    tot_dist = 0
    for i, drone in enumerate(drones):
        tot_dist += drone.current_distance
        print(f"{i + 1}号无人机配送情况: 配送中心: {drone.home_distribution_center}, "
              f"路径: {drone.path}, 路径长度: {drone.current_distance}, 物品数: {drone.current_load}")

    return tot_dist

def main():
    all_orders = []     # 未解决的订单
    total_distance = 0  # 总路径长度
    current_time = 0    # 当前时刻

    while current_time < total_time:
        # 每隔t分钟生成一次订单, 进行一次决策
        new_orders = generate_orders(current_time)
        all_orders.extend(new_orders)

        total_distance += greedy_delivery(all_orders, current_time)

        # 移除已经完成的订单
        all_orders = [order for order in all_orders if order.flag == False]
        current_time += t

    print(f"\n总路径长度: {total_distance}")

if __name__ == '__main__':
    main()