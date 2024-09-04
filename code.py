import random
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# 参数初始化
num_delivery_centers = 5
num_drop_points = 3
max_carry_items = 10
max_distance = 20 * 1000  # 公里转化为米
drone_speed = 60 * 1000 / 60  # 公里/小时转化为米/分钟
time_interval = 30  # 分钟
# 计算距离函数
def calculate_distance(loc1, loc2):
    distance = int(((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5 * 100)  # 欧几里得距离，转化为米
    # print(f"从 {loc1} 到 {loc2} 的距离是：{distance} 米")
    return distance
distances = []
# 生成配送中心和卸货点位置
delivery_centers = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_delivery_centers)]
drop_points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_drop_points)]
for center in delivery_centers:
    for drop_point in drop_points:
        distance = calculate_distance(center, drop_point)
        distances.append(distance)

plt.hist(distances, bins=20)
plt.xlabel('Distance (m)')
plt.ylabel('Frequency')
plt.title('Distribution of Distances')
plt.show()

# 打印配送中心和卸货点位置以便调试
print("配送中心位置:", delivery_centers)
print("卸货点位置:", drop_points)


# 初始化订单信息
orders = []

# 订单生成函数
def generate_orders(current_time):
    new_orders = []
    for drop_point in drop_points:
        num_orders = random.randint(0, 3)
        for _ in range(num_orders):
            priority = random.choice(['一般', '较紧急', '紧急'])
            new_orders.append({
                'location': drop_point,
                'priority': priority,
                'time_created': current_time,
            })
    return new_orders

# 路径规划
def plan_paths(orders):
    num_locations = len(orders) + num_delivery_centers
    starts = [i for i in range(num_delivery_centers)]
    ends = [i for i in range(num_delivery_centers)]
    
    # 创建路由管理器
    manager = pywrapcp.RoutingIndexManager(num_locations, num_delivery_centers, starts, ends)

    routing = pywrapcp.RoutingModel(manager)

    # 定义距离函数
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node < num_delivery_centers:
            from_loc = delivery_centers[from_node]
        else:
            from_loc = orders[from_node - num_delivery_centers]['location']
        if to_node < num_delivery_centers:
            to_loc = delivery_centers[to_node]
        else:
            to_loc = orders[to_node - num_delivery_centers]['location']
        return calculate_distance(from_loc, to_loc)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 添加容量约束
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node < num_delivery_centers:
            return 0
        return 1

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [max_carry_items] * num_delivery_centers,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # 添加距离约束
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        max_distance,
        True,
        'Distance')
    distance_dimension = routing.GetDimensionOrDie('Distance')
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # 设置搜索参数并解决
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 30  # 增加时间限制，提供更多求解时间

    solution = routing.SolveWithParameters(search_parameters)

    # 输出路径
    if solution:
        routes = []
        for vehicle_id in range(num_delivery_centers):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index < num_delivery_centers:
                    route.append(delivery_centers[node_index])
                else:
                    route.append(orders[node_index - num_delivery_centers]['location'])
                index = solution.Value(routing.NextVar(index))
            routes.append(route)
        return routes
    else:
        print('No solution found!')
        return None

# 可视化路径
def plot_routes(routes):
    colors = ['r', 'g', 'b']
    for i, route in enumerate(routes):
        x_coords = [loc[0] for loc in route]
        y_coords = [loc[1] for loc in route]
        plt.plot(x_coords, y_coords, marker='o', color=colors[i % len(colors)], label=f'Drone {i+1}')
    for center in delivery_centers:
        plt.scatter(center[0], center[1], c='black', marker='x', s=100)
    for drop_point in drop_points:
        plt.scatter(drop_point[0], drop_point[1], c='blue', marker='o', s=50)
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Drone Delivery Routes')
    plt.show()

# 主循环
current_time = 0
while current_time < 24 * 60:  # 模拟一天的时间
    print(f'当前时间: {current_time} 分钟')
    new_orders = generate_orders(current_time)
    orders.extend(new_orders)
    print(f'新生成订单: {new_orders}')
    
    # 处理紧急订单
    urgent_orders = [order for order in orders if order['priority'] in ['较紧急', '紧急']]
    if not urgent_orders:
        urgent_orders = orders  # 如果没有紧急订单，则处理所有订单

    routes = plan_paths(urgent_orders)
    if routes:
        print(f'路径: {routes}')
        plot_routes(routes)
        
        # 移除已配送的订单
        orders = [order for order in orders if order not in urgent_orders]

    current_time += time_interval