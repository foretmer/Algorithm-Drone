from config import *
from instance_set import DP_Points
from utils import duplicate_removal

distance_dp2dp = distance_dp2dp
distance_dy2dp = distance_dy2dp
graph = graph
routes_sat = routes_sat
routes_len = routes_len

dp_points = DP_Points(DP_position)
now_time = 0
time_scale = 60 * Time_to_Work  # 规划12小时内的决策 并让订单在第9小时停止完成 让12小时内可以完成所有订单
route_in_12 = []
while now_time <= time_scale:
    if now_time <= time_scale - 60 * 3:  # 小于截止时间前3小时均可以生成订单
        dp_points.generate_new_orders(now_time)
    dp_points.display_all_orders(now_time)
    order_urgent_list = []  # 存储当前决策轮次必须处理的订单数目
    dp_ls = dp_points.dp_point_list
    for dp_i in range(len(dp_ls)):
        dp_order = dp_ls[dp_i].current_orders
        dp_new_order = [x for x in dp_order if x - now_time > 10]
        num = len(dp_order) - len(dp_new_order)
        if num == 0:
            continue
        order_urgent_list.append([dp_list[dp_i], num])  # 当前轮次最紧急的订单先处理
        dp_ls[dp_i].current_orders = dp_new_order
    order_urgent_list = sorted(order_urgent_list, key=lambda x: x[1])  # 先处理需求量大的，让无人机尽可能满载
    print('order_urgent_list: ', order_urgent_list)
    route_now_time = []
    while len(order_urgent_list) != 0:
        # 对于订单量划分两部分
        # 第一部分为专用航线仅飞往目的地的
        order_urgent_max = order_urgent_list.pop()   # 接收的是索引,无参数返回最后一个元素，即最多订单的卸货点
        dp, order_num = order_urgent_max[0], order_urgent_max[1]
        drone_num = order_num // Max_Drone_Carry  # 专航两点直飞需要的无人机数量
        dy2dp_direct_now = []
        for tmp in dy2dp_direct:
            if dp == tmp[0]:
                dy2dp_direct_now = [tmp[1], tmp[2]]
        if drone_num != 0:
            route_now_time.append([drone_num, dy2dp_direct_now[0], dy2dp_direct_now[1]])  # 无人机数目 一趟路径 路径具体信息
        # 第二部分为无人机未实现满载时
        drone_remainder_num = order_num % Max_Drone_Carry  # 剩余的order数目
        if drone_remainder_num == 0:
            continue
        if len(order_urgent_list) != 0:
            dps_dp = list(filter(lambda x: dp == x[0], dps))
            if len(dps_dp) != 0:  # 存在其他路径
                for dps_dp_info in dps_dp:  # 按照路径长短遍历路径寻找
                    # 找出该条路径经过的节点
                    dps_dp_route = list(filter(lambda x: 'dp' in x, dps_dp_info[1])).remove(dp)
                    order_urgent_dp = [x[0] for x in order_urgent_list]
                    dp_to_expand = duplicate_removal(dps_dp_route, order_urgent_dp)
                    if len(dp_to_expand) == 0:  # 若该路径与urgent相交为空 则换下一条路径
                        continue
                    else:
                        while len(dp_to_expand) != 0:
                            dp_exp = dp_to_expand.pop()  # 'dpX'
                            ind = order_urgent_dp.index(dp_exp)
                            order_info = order_urgent_list[ind]
                            if order_info[1] + drone_remainder_num > Max_Drone_Carry:
                                # 扣除需要的订单并更新urgent
                                order_urgent_list[ind][1] = order_info[1] - (Max_Drone_Carry - drone_remainder_num)
                                drone_remainder_num = Max_Drone_Carry
                                break
                            elif order_info[1] + drone_remainder_num == Max_Drone_Carry:
                                # 删除相应的订单并更新urgent
                                drone_remainder_num = Max_Drone_Carry
                                order_urgent_list.pop(ind)
                                break
                            elif order_info[1] + drone_remainder_num < Max_Drone_Carry:
                                # 更新飞机上的订单数量drone_remainder_num 并进入下一轮循环
                                drone_remainder_num = order_info[1] + drone_remainder_num
                                order_urgent_list.pop(ind)
                    if drone_remainder_num <= Max_Drone_Carry:  # 飞机已经满载 找到路径添加到route记录即可
                        need_to_pop = Max_Drone_Carry - drone_remainder_num
                        if need_to_pop != 0:
                            dp_orders = dp_points.dp_point_list[dp_list.index(dp)].current_orders
                            if len(dp_orders) <= need_to_pop:
                                dp_points.dp_point_list[dp_list.index(dp)].current_orders = []
                            else:
                                dp_points.dp_point_list[dp_list.index(dp)].current_orders = dp_orders[need_to_pop:-1]
                        route_now_time.append([1, dps_dp_info[2][-1], dps_dp_info[1]])  # 无人机数目 路径 路径具体信息
                    break  # 执行至此 路径已经锁死 没必要继续遍历路径
            else:  # == 不存在其他路径则调用下一轮次的同一地点的货物使其飞机满载
                need_to_pop = Max_Drone_Carry - drone_remainder_num
                if need_to_pop != 0:
                    dp_orders = dp_points.dp_point_list[dp_list.index(dp)].current_orders
                    if len(dp_orders) <= need_to_pop:
                        dp_points.dp_point_list[dp_list.index(dp)].current_orders = []
                    else:
                        dp_points.dp_point_list[dp_list.index(dp)].current_orders = dp_orders[need_to_pop:-1]
                route_now_time.append([1, dy2dp_direct_now[0], dy2dp_direct_now[1]])  # 无人机数目 路径 路径具体信息
        else:  # == 不存在紧急订单则调用下一轮次的同一地点的货物使其飞机满载
            need_to_pop = Max_Drone_Carry - drone_remainder_num
            if need_to_pop != 0:
                dp_orders = dp_points.dp_point_list[dp_list.index(dp)].current_orders
                if len(dp_orders) <= need_to_pop:
                    dp_points.dp_point_list[dp_list.index(dp)].current_orders = []
                else:
                    dp_points.dp_point_list[dp_list.index(dp)].current_orders = dp_orders[need_to_pop:-1]
            route_now_time.append([1, dy2dp_direct_now[0], dy2dp_direct_now[1]])  # 无人机数目 路径 路径具体信息
    route_in_12.append(route_now_time)
    now_time = now_time + Decision_Time_Interval
routes_res = 0
for route_t in route_in_12:
    if len(route_t) != 0:
        for route in route_t:
            routes_res = routes_res + route[0] * route[1]
print(route_in_12)
print(routes_res)
