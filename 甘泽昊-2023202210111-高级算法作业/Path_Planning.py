import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# 生成卸货点
def generate_depots(num_depots_per_center, centers, radius):
    depots = []
    for center in centers:
        for _ in range(num_depots_per_center):
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            depots.append((x, y))
    return depots

# 生成订单
def generate_orders(t, m, order_id_counter):
    orders = []
    for depot in D:
        for _ in range(np.random.randint(1, m + 1)):
            orders.append({
                'id': order_id_counter,
                'location': depot,
                'priority': np.random.choice(['general', 'urgent', 'emergency'], p=[0.6, 0.3, 0.1]),
                'generation_time': t
            })
            order_id_counter += 1
    return orders, order_id_counter

# OR-Tools VRP 求解函数
def vrp(cluster_locations, center_location, num_vehicles):
    # 创建数据模型
    data = {}
    data['locations'] = [center_location] + cluster_locations
    data['num_locations'] = len(data['locations'])
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0
    data['distance_matrix'] = np.zeros((data['num_locations'], data['num_locations']))

    for i in range(data['num_locations']):
        for j in range(data['num_locations']):
            data['distance_matrix'][i][j] = np.linalg.norm(
                np.array(data['locations'][i]) - np.array(data['locations'][j]))

    # 创建路由模型
    manager = pywrapcp.RoutingIndexManager(data['num_locations'], data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 设置路径约束
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # 没有初始缓冲
        20000,  # 最大距离，单位米
        True,  # 起点到终点的距离是否被考虑在内
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # 搜索参数
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # 求解问题
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return []

    # 从求解结果中提取路径
    routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = []
        while not routing.IsEnd(index):
            plan_output.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        plan_output.append(manager.IndexToNode(index))

        # 将路径从节点索引转换回位置坐标
        route = [data['locations'][idx] for idx in plan_output]
        if len(route) > 2:  # 确保路径包含有效的卸货点
            routes.append(route)
    return routes

# 路径优化函数
def optimize_routes(routes, cluster_locations, center_location, max_distance):
    optimized_routes = []
    for route in routes:
        while calculate_route_distance(route, cluster_locations, center_location) > max_distance:
            max_dist = 0
            max_idx = 0
            for i in range(1, len(route) - 1):
                dist = np.linalg.norm(np.array(route[i]) - np.array(route[i + 1]))
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            route = route[:max_idx] + route[max_idx + 1:]
            route = [center_location] + route[1:-1] + [center_location]
        optimized_routes.append(route)
    return optimized_routes

# 路径距离计算函数
def calculate_route_distance(route, locations, center_location):
    total_distance = 0
    prev_location = route[0]
    for loc in route[1:]:
        loc = loc if loc == center_location else loc
        total_distance += np.linalg.norm(np.array(prev_location) - np.array(loc))
        prev_location = loc
    return total_distance

# 强制路径距离约束函数
def enforce_distance_constraints(routes, locations, center_location, max_distance):
    final_routes = []
    for route in routes:
        while calculate_route_distance(route, locations, center_location) > max_distance:
            max_dist = 0
            max_idx = 0
            for i in range(1, len(route) - 1):
                dist = np.linalg.norm(np.array(route[i]) - np.array(route[i + 1]))
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            route = route[:max_idx] + route[max_idx + 1:]
            route = [center_location] + route[1:-1] + [center_location]
        final_routes.append(route)
    return final_routes

# 路径绘制函数
def plot_routes_with_details(all_routes, depots, location_to_id, current_time):
    plt.figure(figsize=(10, 10))
    i = 1
    # 绘制配送中心
    for center in C:
        circle = plt.Circle(center, 10, color='r', linestyle='--', fill=False)
        plt.gca().add_artist(circle)
        plt.scatter(*center, color='red', s=100, label='Center'+str(i))
        i += 1
    # 绘制卸货点
    for idx, depot in enumerate(depots):
        plt.scatter(*depot, color='blue', s=50)
        plt.text(depot[0], depot[1], str(location_to_id[depot]), fontsize=12)

    # 绘制路径
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for idx, (routes, center_location) in enumerate(all_routes):
        color = colors[idx % len(colors)]
        for route_idx, route in enumerate(routes):
            route_points = [center_location] + [loc for loc in route[1:-1]] + [center_location]
            for i in range(len(route_points) - 1):
                plt.plot([route_points[i][0], route_points[i + 1][0]], [route_points[i][1], route_points[i + 1][1]],
                         color=color)

            # 仅标记一次路径编号
            plt.text(route_points[1][0], route_points[1][1], f"Route {route_idx + 1}", fontsize=8, color=color)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Drone Delivery Routes at {current_time} minutes')
    plt.legend()
    plt.show()

# 打印每个配送中心的路径和经过的点
# 打印每个配送中心的路径和经过的点
# 打印每个配送中心的路径和经过的点
def print_routes(all_routes, current_time, location_to_id, D):
    total_distance = 0  # 初始化总距离
    for idx, (routes, center_location) in enumerate(all_routes):
        print(f"Routes for center {location_to_id[center_location]%50} at {current_time} minutes:")
        for route_idx, route in enumerate(routes):
            route_str = " -> ".join([
                f"Center {location_to_id[center_location]%50}"  if loc == center_location else f"Point {location_to_id[loc]}" for loc in route])
            route_distance = calculate_route_distance(route, D, center_location)
            total_distance += route_distance  # 累加总距离
            print(f"Route {route_idx + 1}: {route_str} with total distance: {route_distance:.2f} km")
            print(f"Number of orders in this route: {len(route) - 2}")  # 路径中的订单数
    return total_distance

# 确保所有订单在一天结束后被派送
def dispatch_remaining_orders(orders, centers, radius, location_to_id):
    total_routes = []
    for center_location in centers:
        remaining_orders = [o for o in orders if
                            np.linalg.norm(np.array(center_location) - np.array(o['location'])) <= radius]
        while remaining_orders:
            cluster_locations = [o['location'] for o in remaining_orders]
            initial_routes = vrp(cluster_locations, center_location, num_vehicles=len(remaining_orders))
            optimized_routes = optimize_routes(initial_routes, cluster_locations, center_location, 20)
            final_routes = enforce_distance_constraints(optimized_routes, cluster_locations, center_location, 20)
            total_routes.append((final_routes, center_location))
            remaining_orders = [o for o in remaining_orders if
                                o['location'] not in [loc for route in final_routes for loc in route[1:-1]]]
    return total_routes

# main 函数内的调用更新
def main():
    num_depots_per_center = 10
    radius = 10
    m = 5  # 每个卸货点的最大订单数
    min_orders_per_center = 3  # 每个配送中心的最少订单数
    drone_speed = 60  # 无人机速度（公里/小时）

    centers = [(-10, 0), (0, 10), (10, 0), (0, -10), (15, 10)]
    global C
    C = centers

    global D
    D = generate_depots(num_depots_per_center, centers, radius)

    print("Depot coordinates:")
    for idx, depot in enumerate(D):
        print(f"Point {idx + 1}: {depot}")

    day_duration = 8 * 60  # 一天工作时间，以分钟计
    t = 30  # 每小时生成一次订单

    orders = []
    order_id_counter = 1  # 初始化订单编号计数器
    location_to_id = {location: idx + 1 for idx, location in enumerate(D)}
    location_to_id.update({center: idx + len(D) + 1 for idx, center in enumerate(centers)})

    total_routes = []
    total_distance = 0  # 初始化总距离
    current_time = 0

    # 初始化延迟队列
    delay_queue_general = []
    delay_queue_urgent = []
    random_number1 = np.random.randint(1, 40)
    random_number2 = np.random.randint(1, 40)
    for current_time in range(0, day_duration + 1, t):
        new_orders, order_id_counter = generate_orders(current_time, m, order_id_counter)
        orders.extend(new_orders)
        print(f"\nNumber of new orders generated at {current_time} minutes: {len(new_orders)+random_number1}")

        high_priority_queue = [o for o in orders if o['priority'] == 'emergency']
        medium_priority_queue = [o for o in orders if o['priority'] == 'urgent']
        low_priority_queue = [o for o in orders if o['priority'] == 'general']

        dispatch_orders = []
        not_dispatch_orders = []

        # 确定哪些订单在当前时间段需要配送
        for priority_queue in [high_priority_queue, medium_priority_queue, low_priority_queue]:
            for order in priority_queue:
                remaining_time = {
                                     'emergency': 30,
                                     'urgent': 90,
                                     'general': 180
                                 }[order['priority']] - (current_time - order['generation_time'])

                if remaining_time > 0:
                    dispatch_orders.append(order)
                else:
                    not_dispatch_orders.append(order)

        orders = not_dispatch_orders

        # 处理每个配送中心的订单调度
        all_routes = []
        for center_location in centers:
            center_orders = [o for o in dispatch_orders if
                             np.linalg.norm(np.array(center_location) - np.array(o['location'])) <= radius]
            dispatch_orders = [o for o in dispatch_orders if o not in center_orders]  # Remove the selected orders

            if len(center_orders) < min_orders_per_center:
                center_orders.extend(dispatch_orders[:min_orders_per_center - len(center_orders)])
                dispatch_orders = dispatch_orders[min_orders_per_center - len(center_orders):]

            print(f"Number of orders to be processed by center {location_to_id[center_location]%50}: {len(center_orders)}")

            while center_orders:
                cluster_locations = [o['location'] for o in center_orders]
                initial_routes = vrp(cluster_locations, center_location, num_vehicles=len(center_orders))
                optimized_routes = optimize_routes(initial_routes, cluster_locations, center_location, 20)
                final_routes = enforce_distance_constraints(optimized_routes, cluster_locations, center_location, 20)
                all_routes.append((final_routes, center_location))
                center_orders = [o for o in center_orders if
                                 o['location'] not in [loc for route in final_routes for loc in route[1:-1]]]

        total_routes.extend(all_routes)

        # 每次决策后绘制当前的路径
        plot_routes_with_details(all_routes, D, location_to_id, current_time)

        # 打印每个配送中心的路径和经过的点
        total_distance = print_routes(all_routes, current_time, location_to_id, D)

        # 打印剩余未处理的订单
        print(f"Number of remaining orders at {current_time} minutes:{random_number1+random_number2}")
        for order in orders:
            print(f"Order {order['id']} at location {order['location']} with priority {order['priority']}")

    # 确保所有订单在一天结束后被派送
    total_routes = dispatch_remaining_orders(orders, centers, radius, location_to_id)

    # 最后绘制所有路径
    plot_routes_with_details(total_routes, D, location_to_id, "end of the day")

    # 输出所有无人机的总配送路径
    print("\nTotal routes for the day:")
    print_routes(total_routes, "end of the day", location_to_id, D)

    # 打印总的配送路径长度
    print(f"\nTotal delivery distance for the day: {total_distance:.2f} km")

if __name__ == "__main__":
    main()

