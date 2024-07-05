import utils
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

INF = 99999

class SA:
    def __init__(self, round, count,  distance_weight, balance_weight,max_single_path_length=INF,speed=60):
        self.T_begin = 200  # 初始温度
        self.T_end = 0.1  # 终止温度
        self.T = self.T_begin  # 过程中的温度, 初始时候是T_begin
        self.T_list = []  # 退火过程中温度列表
        self.Lk = 300  # 每个温度下的迭代次数
        self.alpha = 0.95  # 温度衰减系数
        
        self.per_iter_solution = []  # 每个温度下最优解
        self.all_per_iter_solution = []  # 记录每个温度下每代最优解变化情况

        self.best_solution = []  # 全局最优解
        self.all_best_solution = []  # 记录每个温度下全局最优解
        self.best_solution_multi = []
        
        self.swap_solu_prob = 0.1  # 执行交换产生新解概率
        self.reverse_solu_prob = 0.4  # 执行逆转产生新解的概率
        self.shift_solu_prob = 1 - self.reverse_solu_prob - self.swap_solu_prob  # 执行移位产生新解的概率

        self.distance_weight = distance_weight  # 总路程权重
        self.balance_weight = balance_weight  # 均衡度数权重

        self.start_index = 1  # 起始点也是终点的序号
        self.max_single_path_length = max_single_path_length  # 单个无人机的最长路径
        self.speed = speed # 无人机的速度
        self.cap_num = 30 # 一个无人机单次可以携带的物品数量
        
        # 优先级从0-2，数越小优先级越高
        self.priority = [0.5,1.5,3] # 优先级对应的时效
        self.priority_represent = ['*','d','^'] # 画图时优先级的形状
        self.priority_color = ['firebrick','orange','seagreen'] # 优先级颜色
        self.priority_size = [8,6,4] # 优先级形状大小

        self.current_delivery_orders = []  # 当前轮次需要配送的订单
        self.waiting_delivery_orders = []  # 当前轮次无需配送的订单
        self.orders = []
        
        self.drone_num = 0 # 无人机数目
        self.order_num = 0 # 当前订单数目，第一个为配送中心
        self.dummy_points = [] # 虚点，即起点穿插入订单信息中
        self.solution_len = 0 # 最终的解的长度
        
        self.round = round # 共进行多少轮产生订单
        self.count = count # 每次随机产生多少订单，即卸货点数目
        self.order_interval = 0.5 # 每隔t间隔生成一批订单，小时


    def order_init(self, orders: list) -> list[int]:
        '''
        订单初始化,初始化温度参数、订单和无人机信息,生成全局最优解,该解默认初始化为一对一配送
        Args:
            orders: 本次需要配送的订单信息
        Returns:
            init_sol: 一对一配送的初始解
        '''
        self.current_delivery_orders = orders
        self.drone_num = len(orders)-1
        self.order_num = len(orders)-1
        self.dummy_points = [x for x in range(self.order_num*5 + 1, self.order_num*5 + self.drone_num)]
        self.solution_len = self.drone_num + self.order_num - 2
        self.T = self.T_begin  # 过程中的温度, 初始时候是T_begin
        self.T_list = []
        
        # 注意: 卸货点起点从1开始, 而不是从0
        solution = [x for x in range(1, self.order_num + 1)]

        # 把起始点剔除
        solution.remove(self.start_index)
        
        # 多个无人机, 增加drone_num-1个虚点
        init_sol = [elem for pair in zip(solution, self.dummy_points) for elem in pair]
        
        # 初始化全局最优解和每个温度下最优解
        self.best_solution = self.per_iter_solution = init_sol
        # exit()
        return init_sol

    def get_delivery_distance(self, solution: list)-> tuple[list, list]:
        '''
        获取当前解的距离，返回得到每个具体路径和该路径对应的长度
        Args:
            solution: 解
        Returns:
            all_routines: 该解中对应的具体路径
            routines_dis: 该解的路径对应长度
        '''
        tmp_solu = solution[:]
        
        # 将增加的虚点还原成起始点
        for i in range(self.solution_len):
            if solution[i] in self.dummy_points:
                tmp_solu[i] = self.start_index
                
        # 根据起始点把chrom分成多段
        one_routine = []  # 一个无人机路线, 可以为空
        all_routines = []  # 所有无人机路线
        for v in tmp_solu:
            if v == self.start_index:
                all_routines.append(one_routine)
                one_routine = []
            elif v != self.start_index:
                one_routine.append(v)
                
        # 还有一次需要添加路线
        all_routines.append(one_routine)

        routines_dis = []  # 所有路径总距离组成的列表
        
        # 计算每一条路总的距离
        for r in all_routines:
            distance = 0
            # 有一个无人机路线为空列表, 即一个无人机无需配送
            if len(r) == 0:
                routines_dis.append(distance)
            else:
                distance = self.get_distance_one_path(r)
                routines_dis.append(distance)
        return all_routines, routines_dis

    def get_distance_one_path(self, path: list) -> float:
        '''
        获取当前路径的距离，注意此路径中并不包含起点和终点
        Args:
            path: 路径
        Returns:
            distance: 该路径对应的长度
        '''
        r_len = len(path)
        distance = 0
        for i in range(r_len):
            # 别忘了最后加上起始点到第一个点的距离
            if i == 0:
                distance += utils.get_dis(self.current_delivery_orders[self.start_index],
                                                self.current_delivery_orders[path[i]])
            if i + 1 < r_len:
                self.current_delivery_orders[path[i + 1]]
                distance += utils.get_dis(self.current_delivery_orders[path[i]], self.current_delivery_orders[path[i + 1]])
             # 最后一个顶点, 下一站是起始点
            elif i == r_len - 1:
                distance += utils.get_dis(self.current_delivery_orders[path[i]],
                                                self.current_delivery_orders[self.start_index])
        
        return distance

    def get_cap_one_path(self, path: list) -> int:
        '''
        获取当前路径的距离，注意此路径中并不包含起点和终点
        Args:
            path: 路径
        Returns:
            num: 该路径对应的容量
        '''
        r_len = len(path)
        num = 0
        for i in range(r_len):
            num += self.current_delivery_orders[path[i]][4]
        return num

    def check_priority_cap(self, routines: list) -> bool:
        '''
        检查当前所有路径中是否都满足优先度需求和无人机最大携带容量
        Args:
            path: 路径
        Returns:
            distance: 该路径对应的长度
        '''
        for r in routines:
            if len(r) == 0:
                continue
            dis = 0
            num = 0
            for i in range(len(r)):
                num += self.current_delivery_orders[r[i]][4]
                if i == 0:
                    dis += utils.get_dis(self.current_delivery_orders[self.start_index],
                                                        self.current_delivery_orders[r[i]])
                elif i + 1 < len(r):
                    dis += utils.get_dis(self.current_delivery_orders[r[i+1]],
                                                        self.current_delivery_orders[r[i]])
                if dis/self.speed > self.priority[self.current_delivery_orders[r[i]][3]] or num>self.cap_num:
                    return False
        return True
        
    def obj_func(self, solution: list) -> list:
        """
        计算解的目标函数值, 同时检查每个无人机的路径长度、优先级和容量是否满足约束
        目标函数 Z = distance_weight*总路程 + balance_weight*均衡度
        均衡度 = (max(l)-min(l))/ max(l)
        Args:
            solution: 解
        Returns:
            obj: 解的目标函数值, 或若违反约束则返回极大值
        """
        all_routines, routines_dis = self.get_delivery_distance(solution)
        # 检查每条路径是否超出最大长度约束
        for single_path_length in routines_dis:
            if single_path_length > self.max_single_path_length:
                # 如果任何路径长度超过限制, 返回一个极大值作为惩罚
                return INF
        
        if not self.check_priority_cap(all_routines):
            return INF
        
        sum_path = sum(routines_dis)
        max_path = max(routines_dis)
        min_path = min(routines_dis)
        if max_path == 0:  # 避免除以零的情况
            balance = 0
        else:
            balance = (max_path - min_path) / max_path
        obj = self.distance_weight * sum_path + self.balance_weight * balance
        return obj

    def swap_solution(self, solution):
        """
        交换产生新解, 与交换变异类似
        Args:
            solution: 解

        Returns:
            new_solution: 新解
        """
        # 如果index1和index2相等, 则交换变异相当于没有执行
        index1 = random.randint(0, self.solution_len - 1)
        index2 = random.randint(0, self.solution_len - 1)
        new_solution = solution[:]
        new_solution[index1], new_solution[index2] = new_solution[index2], new_solution[index1]
        return new_solution

    def shift_solution(self, solution: list) -> list:
        """
        移位产生新解: 随机选取三个点, 将前两个点之间的点移位到第三个点的后方
        solution     = [5, 8, 6, 1, 12, 4, 11, 15, 2]
        new_solution = [5, 4, 11, 8, 6, 1, 12, 15, 2]
        Args:
            solution:解

        Returns:
            new_solution:新解
        """
        tmp = sorted(random.sample(range(self.solution_len), 3))  # 随机选取3个不同的数
        index1, index2, index3 = tmp[0], tmp[1], tmp[2]
        tmp = solution[index1:index2]
        new_solution = []
        for i in range(self.solution_len):
            if index1 <= i < index2:
                continue
            if (i < index1 or i >= index2) and i < index3:
                new_solution.append(solution[i])
            elif i == index3:
                new_solution.append(solution[i])
                new_solution.extend(tmp)
            else:
                new_solution.append(solution[i])
        return new_solution

    def reverse_solution(self, solution: list) -> list:
        """
        逆转: 随机选择两点(可能为同一点), 逆转其中所有的元素
        solution     = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        new_solution = [1, 2, 6, 5, 4, 3, 7, 8, 9]
        Args:
            solution:父代

        Returns:
            new_solution: 逆转变异后的子代
        """
        index1, index2 = random.randint(0, self.solution_len - 1), random.randint(0, self.solution_len - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        new_solution = solution[:]
        tmp = new_solution[index1: index2]
        tmp.reverse()
        new_solution[index1: index2] = tmp
        return new_solution

    def generate_new_solu(self, solution: list) -> list:
        """
        产生新解的过程类似变异过程
        Args:
            solution: 解

        Returns:
            new_solution: 新解
        """
        """
        prob_sum表示一种累加和的列表, 比如: 
        四种变异可能性[0.2, 0.3, 0.4, 0.1]
        prob_sum = [0.2, 0.5, 0.9, 1]
        变异只有三种变异, 这里采用了硬编码
        """
        prob_sum = []
        prob_sum.extend([self.swap_solu_prob, self.swap_solu_prob + self.reverse_solu_prob, 1])
        p = random.random()
        if p <= prob_sum[0]:
            # 交换产生新解
            new_solution = self.swap_solution(solution)
        elif p <= prob_sum[1]:
            # 逆转产生新解
            new_solution = self.reverse_solution(solution)
        else:
            # 移位产生新解
            new_solution = self.shift_solution(solution)
        return new_solution

    def print_best_routine(self):
        """
        打印无人机最优配送路线
        """
        best_path, best_dist_list = self.get_delivery_distance(self.best_solution)
        print("无人机所有路线长度为: {}".format(sum(best_dist_list)))
        count = 1
        
        # 打印全局最优个体的所有路线卸货点（包括起点和终点）
        for i in range(len(best_path)):
            if best_dist_list[i] ==0:
                continue
            print("第{}架无人机路线长度为: {},需要经过{}个卸货点,携带{}件货物，".format(count, best_dist_list[i],len(best_path[i]),self.get_cap_one_path(best_path[i])))
            print("第{}架无人机路线为: ".format(count), end="")
            if len(best_path[i]) != 0:
                dis = 0
                num = 0
                count += 1
                for j in range(len(best_path[i])):
                    num += self.current_delivery_orders[best_path[i][j]][4]
                    if j == 0:
                        dis += utils.get_dis(self.current_delivery_orders[self.start_index],
                                                  self.current_delivery_orders[best_path[i][j]])
                        print("{} ——> {} ".format(self.current_delivery_orders[self.start_index][0],
                                                  self.current_delivery_orders[best_path[i][j]][0]), end="")
                    if j + 1 < len(best_path[i]):
                        dis += utils.get_dis(self.current_delivery_orders[best_path[i][j + 1]],
                                                  self.current_delivery_orders[best_path[i][j]])
                        print("——> {} ".format(self.current_delivery_orders[best_path[i][j + 1]][0]), end="")
                    elif j == len(best_path[i]) - 1:
                        print("——> {} ".format(self.current_delivery_orders[self.start_index][0]))
            
    def gen_random_order(self, count: int, ts: int = 0) -> list:
        """
        产生随机订单
        Args:
            count: 产生订单的数量
            ts: 这一批订单的时间戳
        Returns:
            orders: 生成的随机订单
        """
        orders = [[]]
        orders.append(["s",7.0,7.0,0,0])
        for i in range(count):
            order = ["o_"+str(i+ts*count)]
            x = 7.0
            y = 7.0
            while x == 7.0 and y== 7.0:
                x = round(random.uniform(0, 14), 2) # 随机生成的二维坐标
                y = round(random.uniform(0, 14), 2)
            order.append(x)
            order.append(y)
            order.append(random.randint(0, 2)) # priority优先级
            order.append(random.randint(1, 5)) # 该订单的货物数量
            order.append(ts) # 该订单生成的时间戳
            order.append(self.priority[order[3]]) # 该订单距离优先级规定的时效还剩余时长
            orders.append(order)
        return orders

    def select_orders(self, orders: list, ts: int) -> list:
        """
        从新生成的订单和之前未配送的订单中选择当前轮应当配送的订单
        Args:
            orders: 本轮产生的订单
            ts: 当前的时间戳
        Returns:
            res_orders: 经过选择的订单
        """
        res_orders = [[],orders[1]]
        orders.pop(0)
        orders.pop(0)
        waiting_orders = []
        
        if ts == self.round-1:
            res_orders.extend(self.waiting_delivery_orders)
            res_orders.extend(orders)
            self.waiting_delivery_orders = []
            return res_orders
        
        # 在之前未配送订单中挑出如果再等待一个t间隔后会超过时效的订单
        for j in range(len(self.waiting_delivery_orders)):
            self.waiting_delivery_orders[j][6] -= self.order_interval
            if (utils.get_dis(res_orders[1],self.waiting_delivery_orders[j])/self.speed) > self.priority[self.waiting_delivery_orders[j][3]]-self.order_interval*(ts-self.waiting_delivery_orders[j][5]):
                res_orders.append(self.waiting_delivery_orders[j])
            else:
                waiting_orders.append(self.waiting_delivery_orders[j])
                
        # 在新生成的订单生成优先级为0(优先级最高)和如果再等待一个t间隔后会超过时效的订单
        for i in range(len(orders)):
            orders[i][6] -= self.order_interval
            if (utils.get_dis(res_orders[1],orders[i])/self.speed) > self.priority[orders[i][3]]-self.order_interval:
                res_orders.append(orders[i])
            else:
                waiting_orders.append(orders[i])
        
        # 为避免订单进行堆积，我们在此规定了每次选择的订单数至少要超过所有订单的30%，可删去
        count = int((len(res_orders)+len(waiting_orders))* 0.3)
        if count <= 0:
            self.waiting_delivery_orders = waiting_orders # 如果已经满足上述条件，则将等待订单列表保存
        else:
            t_orders = sorted(waiting_orders, key=lambda x: (x[6], -x[4])) # 将本轮无需配送的订单按照以剩余时长升序和货物数量降序
            res_orders.extend(t_orders[:count]) # 添加所需差值到当前需要配送的订单列表中
            self.waiting_delivery_orders = t_orders[count:]
        return res_orders

    def replenish(self):
        """
        由于最初选择的订单可能生成的路径中的无人机还拥有携带能力，因此在这里用贪心算法进行选择扩充使无人机尽可能达到满载
        """
        if len(self.waiting_delivery_orders) == 0:
            return
        
        best_path, best_dist_list = self.get_delivery_distance(self.best_solution) # 获取最优解的路径
        self.waiting_delivery_orders.sort(key=lambda x: (x[1], -x[2])) # 将未配送订单按照坐标进行排序
        for i in range(len(best_path)):
            
            # 如果路径为空则跳过
            if best_dist_list[i] == 0:
                continue
            flag = True # flag则为如果在贪心算法中未找到合适值则不对该无人机进行修改
            
            while best_dist_list[i] < self.max_single_path_length * 0.9 and flag : # 最终想要无人机配送路程达到最长路径90%

                point = np.array([(item[1], item[2]) for item in self.waiting_delivery_orders]) # 获取当前未配送订单坐标
                nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree') # 只对每个查询点感兴趣于最近的一个邻居
                nn_model.fit(point)
                
                # 该路径上的坐标
                query_points = []
                for index in range(len(best_path[i])):
                    query_points.append([self.current_delivery_orders[best_path[i][index]][1],
                                            self.current_delivery_orders[best_path[i][index]][2]])
                    
                # 分别获取到离当前最佳路径上每个点的最近点的距离和在waiting中的索引值
                distances, indices = nn_model.kneighbors(query_points)
                
                # 提取最小距离和其在waiting中的索引 
                dis = []
                for k in range(len(distances)):
                    dis.append([distances[k][0],indices[k][0]])
            
                # 按照最小距离进行升序
                dis_sorted = sorted(dis, key=lambda x: x[0])
                j = 0
                
                # 使用贪心算法，从最小的值开始尝试加入路径
                for j in range(len(dis_sorted)):
                    self.current_delivery_orders.append(self.waiting_delivery_orders[dis_sorted[j][1]]) # 将订单加入到当前配送订单中  
                    best_path[i].insert(j+1,len(self.current_delivery_orders)-1) # 将卸货点加入距离最近的路径中卸货点的后面
                    
                    # 如果加入后该路径超过路径最大距离或者优先级和无人机容量不满足约束，则从路径中移除该订单
                    if self.get_distance_one_path(best_path[i]) > self.max_single_path_length or not self.check_priority_cap([best_path[i]]):
                        best_path[i].pop(j+1)
                        best_path[i].insert(j,len(self.current_delivery_orders)-1) # 将从路径中移除的订单加入到距离最近的路径卸货点的前面
                        
                        # 如果仍然不满足三个约束条件，则该订单不符合要求，则从路径和当前配送订单中均删除该订单
                        if self.get_distance_one_path(best_path[i]) > self.max_single_path_length or not self.check_priority_cap([best_path[i]]):
                            self.current_delivery_orders.pop(len(self.current_delivery_orders)-1) 
                            best_path[i].pop(j)
                        else: 
                            # 订单插入对应卸货点前面时满足三个约束条件
                            self.waiting_delivery_orders.pop(dis_sorted[j][1]) # 从等待订单信息中删除该订单
                            
                            # 将该订单加入到最优解中
                            if j == len(dis_sorted)-1:
                                self.best_solution.insert(self.best_solution.index(best_path[i][j-1])+1,best_path[i][j])
                            else :
                                self.best_solution.insert(self.best_solution.index(best_path[i][j+1]),best_path[i][j])
                            self.solution_len += 1 # 修改解的长度值
                            best_dist_list[i] = self.get_distance_one_path(best_path[i]) # 重新获取这个路径长度
                            break
                    else:
                        # 订单插入对应卸货点后面时满足三个约束条件
                        self.waiting_delivery_orders.pop(dis_sorted[j][1]) # 从等待订单信息中删除该订单
                        self.best_solution.insert(self.best_solution.index(best_path[i][j])+1,best_path[i][j+1]) # 将该订单加入到最优解中
                        self.solution_len += 1 # 修改解的长度值
                        best_dist_list[i] = self.get_distance_one_path(best_path[i]) # 重新获取这个路径长度
                        break # 此时退出则可以重新根据修改后的路径继续求最小距离的卸货点列表
                else:
                    flag = False # 只有路径中每个卸货点对应的最小距离加入路径中，均不满足三个约束条件则进行下个路径
    
    def plot_method(self,ts: str = 'final'):
        """
        绘制本轮次的路径选择和未配送订单信息
        Args:
            orders: 本轮产生的订单
            ts: 当前的时间戳
        Returns:
            res_orders: 经过选择的订单
        """
        best_path, best_dist_list = self.get_delivery_distance(self.best_solution)

        # 创建一个新的图和坐标轴
        fig, ax = plt.subplots()
        
        # 绘制起点，即配送中心
        plt.plot(self.current_delivery_orders[self.start_index][1],
                 self.current_delivery_orders[self.start_index][2],"r*", markersize=10)
        
        for i in range(len(best_path)):
            
            if best_dist_list[i] == 0 or len(best_path[i]) == 0:
                continue # 如果路径长度为0或者路径为空则跳过
            else:
                color = utils.generate_random_color() # 随机选择一个颜色，每个不同的颜色则代表一架不同的无人机
                for j in range(len(best_path[i])): 
                    if j == 0:
                        # 绘制起点到路径第一个路径中的卸货点
                        plt.arrow(self.current_delivery_orders[self.start_index][1],
                                  self.current_delivery_orders[self.start_index][2], 
                                  self.current_delivery_orders[best_path[i][j]][1]-self.current_delivery_orders[self.start_index][1], 
                                  self.current_delivery_orders[best_path[i][j]][2]-self.current_delivery_orders[self.start_index][2], # 坐标与距离
                            head_width=0.2, lw=1, # 箭头⻓度，箭尾线宽
                            color=color,length_includes_head = True) # ⻓度计算包含箭头箭尾
                        plt.plot(self.current_delivery_orders[best_path[i][j]][1],
                                self.current_delivery_orders[best_path[i][j]][2],
                                marker = self.priority_represent[self.current_delivery_orders[best_path[i][j]][3]],
                                color = self.priority_color[self.current_delivery_orders[best_path[i][j]][3]],
                                markersize = self.priority_size[self.current_delivery_orders[best_path[i][j]][3]]
                                ) # 绘制第一个路径中的卸货点
                    if j + 1 < len(best_path[i]):
                        plt.arrow(self.current_delivery_orders[best_path[i][j]][1],
                                  self.current_delivery_orders[best_path[i][j]][2], 
                                  self.current_delivery_orders[best_path[i][j + 1]][1]-self.current_delivery_orders[best_path[i][j]][1], 
                                  self.current_delivery_orders[best_path[i][j + 1]][2]-self.current_delivery_orders[best_path[i][j]][2], # 坐标与距离
                            head_width=0.2, lw=1, # 箭头⻓度，箭尾线宽
                            color=color,length_includes_head = True)
                        plt.plot(self.current_delivery_orders[best_path[i][j+1]][1],
                                self.current_delivery_orders[best_path[i][j+1]][2],
                                marker = self.priority_represent[self.current_delivery_orders[best_path[i][j+1]][3]],
                                color = self.priority_color[self.current_delivery_orders[best_path[i][j+1]][3]],
                                markersize=self.priority_size[self.current_delivery_orders[best_path[i][j+1]][3]]
                                ) # 绘制该箭头线条的前一个卸货点
                    elif j == len(best_path[i]) - 1:
                        plt.arrow(self.current_delivery_orders[best_path[i][j]][1],
                                  self.current_delivery_orders[best_path[i][j]][2], 
                                  self.current_delivery_orders[self.start_index][1]-self.current_delivery_orders[best_path[i][j]][1], 
                                  self.current_delivery_orders[self.start_index][2]-self.current_delivery_orders[best_path[i][j]][2], # 坐标与距离
                            head_width=0.2, lw=1, # 箭头⻓度，箭尾线宽
                            color=color,length_includes_head = True) # 绘制最后卸货点到配送中心
        
        # 绘制等待配送的订单信息，颜色为灰色
        for k in range (len(self.waiting_delivery_orders)):
            plt.plot(self.waiting_delivery_orders[k][1],
                self.waiting_delivery_orders[k][2],
                marker = self.priority_represent[self.waiting_delivery_orders[k][3]],
                color = 'grey',
                markersize = self.priority_size[self.waiting_delivery_orders[k][3]]
                )
        
        ax.set_xlim(-1, 15)
        ax.set_ylim(-1, 15)
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 设置标题
        plt.title('Delivery Round '+ts)

        # 绘制图例
        legend_elements = [plt.Line2D([0], [0], marker='*', color='w', label='Priority 1', markerfacecolor='firebrick', markersize=8),
                            plt.Line2D([0], [0], marker='d', color='w', label='Priority 2', markerfacecolor='orange', markersize=6),
                            plt.Line2D([0], [0], marker='^', color='w', label='Priority 3', markerfacecolor='seagreen', markersize=4)]

        # 显示自定义图例
        plt.legend(handles=legend_elements, loc='best')

        # 显示图表
        plt.show()
          
    def sa_process_iterator(self, solution):
        '''
        进行不同温度下的多轮迭代
        '''
        while self.T > self.T_end:
            # 每个温度下最优解都要赋值
            self.per_iter_solution = solution
            # 在每个温度下迭代
            for _ in range(self.Lk):
                # 当前解的目标函数值
                current_solu_obj = self.obj_func(solution)
                # 产生新解
                new_solu = self.generate_new_solu(solution)
                # 新解目标函数值
                new_solu_obj = self.obj_func(new_solu)
                # 新解更优, 接受新解
                if new_solu_obj < current_solu_obj:
                    solution = new_solu
                # Metropolis准则
                elif new_solu_obj != INF:
                # else:
                    prob_accept = math.exp(-(new_solu_obj - current_solu_obj) / self.T)
                    p = random.random()
                    if p < prob_accept:
                        solution = new_solu
            # 该温度下迭代完成
            solu_obj = self.obj_func(solution)
            if solu_obj == INF:
                continue
            # 解和该温度下最优解比较
            if solu_obj < self.obj_func(self.per_iter_solution):
                self.per_iter_solution = solution
            # 解和全局最优解比较
            if solu_obj < self.obj_func(self.best_solution):
                self.best_solution = solution
            # 记录每个温度下最优解和全局最优解
            self.all_per_iter_solution.append(self.per_iter_solution)
            self.all_best_solution.append(self.best_solution)

            # ********************参数打印********************************
            # print("在T = {} 温度下:".format(self.T))
            # print("该温度下, 最优解为{}".format(self.per_iter_solution))
            # print("该温度下, 最优解路线为{}".format(per_iter_solu_path))
            # print("该温度下, 最优解路线长度为{}".format(sum(per_iter_solu_dis)))
            # print("该温度下, 最优解路线长度列表为{}".format(per_iter_solu_dis))
            # print("---------------------------------------------------------")
            # print("全局最优解为{}".format(self.best_solution))
            # print("全局最优解路线为{}".format(best_solu_path))
            # print("全局最优解路线长度为{}".format(sum(best_solu_dis)))
            # print("全局最优解路线长度列表为{}".format(best_solu_dis))
            # print("**************************************************************************")

            # *******************有关参数更新****************************
            self.T_list.append(self.T)
            self.T = self.T * self.alpha

    def sa_process(self):
        """
        根据相关参数，开始求解，主要负责生成随机订单、订单初始化、求得当前订单的最优解、补充路径订单、打印具体信息和绘制示意图
        """
        for i in range(self.round):
            self.orders.append(self.gen_random_order(self.count,i))
            dis = 0
            cur_orders = self.select_orders(self.orders[i],i)
            if len(cur_orders) == 2:
                continue
            self.order_init(cur_orders)
            self.sa_process_iterator(self.per_iter_solution)
            self.replenish()
            for j in range(len(self.current_delivery_orders)-2):
                dis += utils.get_dis(self.current_delivery_orders[1],self.current_delivery_orders[j+2])
            print("第{}轮配送订单{}个，一对一配送的总配送路程为{:.3f}".format(i+1,len(cur_orders)-2,dis))
            self.print_best_routine()
            self.plot_method(str(i+1))
        
if __name__ == "__main__":
    sa_obj = SA(round = 5,
                count = 100,
                distance_weight=1,
                balance_weight=2000,
                max_single_path_length=20)
    sa_obj.sa_process()