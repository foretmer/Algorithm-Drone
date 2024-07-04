# Advanced_Algorithm_VRP(高级算法大作业-无人机配送路径规划问题)

本次作业的文档及代码均已开源至Github，链接：[hi-underworld/Advanced_Algorithm_VRP (github.com)](https://github.com/hi-underworld/Advanced_Algorithm_VRP)。截至6月29日的clone及visit情况如下：

![image-20240629122845391](C:\Users\10715\AppData\Roaming\Typora\typora-user-images\image-20240629122845391.png)

[TOC]

## 1、问题描述

无人机可以快速解决最后10 公里的配送，本作业要求设计一个算法，实现如下图所示区域
的无人机配送的路径规划。在此区域中，共有j 个配送中心，任意一个配送中心有用户所需
要的商品，其数量无限，同时任一配送中心的无人机数量无限。该区域同时有 k 个卸货点
（无人机只需要将货物放到相应的卸货点即可），假设每个卸货点会随机生成订单，一个订
单只有一个商品，但这些订单有优先级别，分为三个优先级别（用户下订单时，会选择优先
级别，优先级别高的付费高）：

- 一般：3 小时内配送到即可； 
- 较紧急：1.5 小时内配送到； 
- 紧急：0.5 小时内配送到。 

将时间离散化，也就是每隔t 分钟，所有的卸货点会生成订单（0-m 个订单），同时每
隔t 分钟，系统要做成决策，包括

- 哪些配送中心出动多少无人机完成哪些订单； 
- 每个无人机的路径规划，即先完成那个订单，再完成哪个订单，...，最后返回原来的配送
  中心；

（ps：系统做决策时，可以不对当前的某些订单进行配送，因为当前某些订单可能紧急程度
不高，可以累积后和后面的订单一起配送。）



## 2、目标

一段时间内（如一天），所有无人机的总配送路径最短



## 3、约束条件

满足订单的优先级别要求



## 4、基本假设

- 无人机一次最多只能携带n 个物品；
- 无人机一次飞行最远路程为20 公里（无人机送完货后需要返回卸货点）； 
- 无人机的速度为60 公里/小时， 即1公里/min；
- 配送中心的无人机数量无限； 
- 任意一个配送中心都能满足用户的订货需求；



## 5、VRP&TWVRP简介

车辆路径规划问题（VRP，Vehicle Routing Problem）是一个经典的组合优化问题，主要目标是为一组车辆设计最优路径，以满足一组客户的需求，同时尽可能减少总行驶距离或总成本。每辆车从一个配送中心出发，服务于若干客户，然后返回配送中心。VRP广泛应用于物流、配送和运输等领域。

时间窗口车辆路径规划问题（TWVRP，Time-Window Vehicle Routing Problem）是VRP的一个变体，增加了时间维度的约束。每个客户有一个指定的时间窗口，车辆必须在这个时间窗口内到达并完成服务。TWVRP进一步增加了问题的复杂性，因为需要同时考虑车辆容量和时间窗口的约束。

#### 发展过程

1. **早期研究（20世纪50年代）**：
   - VRP的研究始于Dantzig和Ramser在1959年提出的卡车调度问题。他们提出了基于线性规划的方法来解决简单的VRP。
2. **经典算法（20世纪60年代-80年代）**：
   - **分支定界法（Branch and Bound）**：一种精确算法，通过构建解的搜索树来逐步逼近最优解。
   - **动态规划（Dynamic Programming）**：Bellman提出了动态规划方法，可以用于求解小规模的VRP。
3. **启发式算法和元启发式算法（20世纪90年代）**：
   - **贪婪算法（Greedy Algorithm）**和**最近邻算法（Nearest Neighbor Algorithm）**：快速生成初始解，但可能陷入局部最优。
   - **模拟退火（Simulated Annealing, SA）**和**遗传算法（Genetic Algorithm, GA）**：通过模拟物理退火过程和生物进化过程来搜索全局最优解。
   - **禁忌搜索（Tabu Search, TS）**：利用禁忌表避免重复搜索，跳出局部最优。
   - **蚁群优化（Ant Colony Optimization, ACO）**：通过模拟蚁群觅食行为，逐步找到最优路径。
4. **现代方法（21世纪）**：
   - **混合算法**：结合多种算法的优点，如遗传算法与局部搜索相结合，提高求解效率。
   - **精英策略**：在元启发式算法中引入精英策略，保留最优解，防止优质解的丢失。
   - **强化学习**：利用深度强化学习来训练模型，逐步逼近最优解。





## 6、解决方案-带时间窗的订单调度及路径规划模型

### 6.1模型概要

整个模型主要分为三个子模型：

1. **卸货点分类模型：**将所有卸货点依据配送中心进行分类。每个`drop\_point_i`有对应的`depot_j`。 所有卸货点为`drop\_point_i`的订单，在后续的订单调度配送的过程中，均由配送中心`depot_j`的无人机进行配送。
2. **订单调度模型：**将处理订单的最小时间间隔设置为`dealing_window`（即订单生成的时间间隔），每个处理订单的时间间隔内，只处理当前时间间隔内必须送出的订单，进而保证所有订单都能在规定的时间窗口内完成。而对于所有必须在第$i$个时间间隔内必须完成的所有订单，采用订单调度模型对这些订单的配送进行调度，并考虑分配哪些订单到一台无人机，使得一个时间窗口内无人机的路径尽可能短进而逼近整个时间周期内无人机的路径最短
3. **车辆最短路径模型**：所有订单完成调度后，对所有分配了订单的车辆进行路径规划。对于类似TSP问题的订单最短配送路径规划，使用遗传算法搜索使得每台无人的路径尽可能短进而逼近一个时间窗口内无人机的路径最短。



### 6.2、卸货点分类模型

卸货点分类模型旨在简化整个问题。由于n个配送中心到m个卸货点的vrp问题相对复杂，因此考虑使用卸货点分类模型，将问题简化为n个（1个配送中心到k个卸货点的vrp问题）。相当于在订单调度开始前，根据配送中心以及卸货点的相对位置确定每个卸货点的订单由固定的配送中心处理。

#### 6.2.1、主要步骤 

1. **初始化**：随机生成的m个卸货点和n个配送中心，即卸货点列表（`drop_points`）和配送中心列表（`depots`），并设置一个阈值（`therehold`）。 根据题目中给出的约束，所有配送中心均能满足所有卸货点的订单，且无人机的运输距离约束为`max_distance = 20`，因此，所有卸货点和所有配送中心的距离`distance`满足约束`distance <= 10`。因此可以考虑以下卸货点和配送中心的随机生成模型。生成一个半径`radius=5`的圆，生成n个配送中心，它们随机分布在这个圆上；在这个圆内随机生成m个卸货点。如下所示：

   ![initial_map.drawio](C:\Users\10715\Desktop\Advanced_Algorithm_VRP\images\initial_map.drawio.svg)

   依据这个规则生成的配送中心和卸货点的分布图，每个配送中心均能满足所有卸货点的订单需求，即`distance [depot_i , drop_point_j]<=10`恒成立。而且这种配送模式也符合当前许多物流公司的配送中心设立，即将大的配送中心分布在城市的周围，在城市中心在设置卸货点。

2. **统计阈值范围内的配送中心**：对于每个卸货点（`drop_point_i`），计算其阈值范围（半径为`therehold`）内其他卸货点的配送中心数量。 如果某个卸货点`drop_point_j`和`drop_point_i`的距离在阈值范围内且已分配了对应的配送中心，则统计该配送中心的数量。 

3. **分配最接近的配送中心**： 如果以`drop_point_i`为圆心，阈值为半径的范围内有卸货点分配了对应的配送中心，则选择数量最多的配送中心分配给当前卸货点`drop_point_i`。如果没有，则选择距离当前卸货点最近的配送中心。下面是一个例子，表示如何为卸货点分配配送中心。灰色的点表示`drop_point_i`，阈值为半径的范围内有5个卸货点，相同的颜色表示被分配到同一个配送中心，不同颜色表示被分配到不同的配送中心。因为4个点有2个红色即两个卸货点被分配到了配送中心`depot_j`，数量是最多的，因此，将`drop_point_i`分配到`depot_j`。

<img src="C:\Users\10715\Desktop\Advanced_Algorithm_VRP\images\knn.png" alt="knn" style="zoom:67%;" />



#### 6.2.2、代码实现

**1、初始化地图**

```python
def initialize_drop_points(num_drop_points: int,max_distance:float) -> List[DropPoint]:
    drop_points = []
    thetas = [random.uniform(0, 2 * math.pi) for _ in range(num_drop_points)]
    distances = [random.uniform(0, max_distance) for _ in range(num_drop_points)]

    drop_point = [(distances[i]* math.cos(thetas[i]), distances[i] * math.sin(thetas[i])) for i in range(len(thetas))]
    for k in range(num_drop_points):
        x = drop_point[k][0]
        y = drop_point[k][1]
        drop_points.append(DropPoint(x, y, k + 1))
    return drop_points

def initialize_depots(num_depots, radium: float) -> List[DePot]:
    thetas = [random.uniform(0, 2 * math.pi) for _ in range(num_depots)]
    depots = []
    depot = [(radium* math.cos(thetas[i]), radium * math.sin(thetas[i])) for i in range(len(thetas))]
    for k in range(num_depots):
        x = depot[k][0]
        y = depot[k][1]
        depots.append(DePot(x, y, k+1))
    return depots
```

**2&3、卸货点分类**

```python
#using the algorithm to find the depot that is closest to the drop point
def find_closest_depot(drop_points: List[DropPoint], depots: List[DePot], therehold: float):
    for drop_point in drop_points:
        # 统计当前drop point半径therehold范围内的drop point的对应的不同depot的数量
        depot_count = {}
        for drop_point_ in drop_points:
            if drop_point_ != drop_point:
                distance = calculate_distance([drop_point.x, drop_point.y], [drop_point_.x, drop_point_.y])
                if distance <= therehold and drop_point_.depot is not None:
                    if drop_point_.depot.id in depot_count:
                        depot_count[drop_point_.depot.id] += 1
                    else:
                        depot_count[drop_point_.depot.id] = 1

        # 找到当前drop point半径therehold范围内的drop point的对应的不同depot的数量最多的depot,并将当前drop point的depot设置为这个depot
        if len(depot_count) != 0:
            max_count = max(depot_count.values())
            for depot_id, count in depot_count.items():
                if count == max_count:
                    drop_point.depot = depots[depot_id-1]
                    print(f"Drop point {drop_point.id} is closest to depot {depot_id}.")
                    break
        else:
            # 如果当前drop point半径therehold范围内没有drop point对应的depot,则将当前drop point的depot设置为距离当前drop point最近的depot
            min_distance = float('inf')
            for depot in depots:
                distance = calculate_distance([drop_point.x, drop_point.y], [depot.x, depot.y])
                if distance < min_distance:
                    min_distance = distance
                    drop_point.depot = depot
            print(f"Drop point {drop_point.id} is closest to depot {drop_point.depot.id}.")

```



### 6.3、订单调度模型

订单调度模型主要是将处理订单的最小时间间隔设置为`dealing_window = m * t`（**t为订单生成的时间间隔**），每个处理订单的时间间隔`dealing_window`内，只处理当前时间间隔内必须送出的订单，进而保证所有订单都能在规定的时间窗口内完成。显然m的值会影响一个`dealing_window`内的订单调度:m的值设置的越大，这个时间间隔内能够处理的调度的订单越多，。而对于所有必须在第`k`个时间间隔内必须完成的所有订单，在进行订单分配。主要为以下几个步骤：

#### 6.3.1、主要步骤

1. **统计最小订单处理时间间隔内应处理的订单**：对于当前待处理的所有订单，计算从其对应的配送中心（`depot`）直接到订单目的地（`drop_point`）的运输时间。 如果订单的截止时间减去运输时间（即订单最晚开始处理时间）首次小于当前处理窗口的结束时间，则认为订单应当在当前订单处理窗口处理。统计所有应当在当前订单处理时间窗口的订单`due_orders`。以下是一个统计订单的例子，

   ![due_order.drawio](C:\Users\10715\Desktop\Advanced_Algorithm_VRP\images\due_order.drawio.svg)

   显然，m值设置有范围，最小显然为1，最大即当前时间`current time = k * t`生成的订单全部都在当前订单处理时间窗`[k * t ,  k * t + dealing_window ]`处理，订单的优先级为30min，60min，180min内完成，因此，`m * t <= 180 - max transport time`,即 `m  <= （180 - max transport time） / t`。因为`distance [depot_i , drop_point_j]<=10`,所以`max transport time <= 10`。

   m过大没有意义，因为超过阈值后，当前订单处理时间窗内包含当前生成的所有订单，因此缺少订单在时间上调度，导致相同卸货点的订单被同一辆无人机运输的概率降低，导致最终总的无人机的路径增大。但如果m过小，导致当前订单处理时间窗内调度的订单过少，进而导致无人机的空载率增大最终导致总的无人机路径增大。因此，订单处理时间窗的大小，即m的值应当根据实际的订单生成数量以及订单生成的时间间隔等实际的一些参数，来决定合适的m值。

   

2. **订单分配初始化**：在统计完当前处理订单的最小时间间隔内的订单`due_orders`后，初始化一个订单配送的方案：将所有`drop_point`即卸货点相同的订单`due_orders_drop_point_i`用`num_vehicles_needed`台无人机运送，`num_vehicles_needed`等于`due_orders_drop_point_i`的重量除以一台无人机的承重`capacity`的值向上取整。最终总是会有小于等于k台无人机没有满载，k为这批订单中不同的卸货点的数量。以下是一个订单分配初始化的一个例子：

   <img src="C:\Users\10715\Desktop\Advanced_Algorithm_VRP\images\initial_plan.drawio.svg" alt="initial_plan.drawio" style="zoom: 70%;" />

   由于步骤1中统计最小订单处理时间间隔内应处理的订单的规则，这样分配的订单一定能够在订单的截止时间前完成。即这种订单分配策略一定合法。

   

3. **针对未满载的无人机进行订单再分配**：对于已经满载的无人机，无需在考虑订单再分配，直接运送至该批订单相同的卸货点即可。而对于未满载的无人机，则需要考虑订单的再分配，即不断将两台无人机的订单进行合并由一台无人机完成配送，直到不存在能够合并订单的两台无人机（可能的原因包括无人机满载，无人机运输单次里程达到上限，无人机无法满足订单的截止时间）。这可以降低这批订单无人机的里程，可以通过简单的数学来证明。

   假设现在有两台无人未满载。由于订单初始化的规则，一定不会存在两台无人机，他们装载了卸货点相同的订单。现在无人机`vehicle_i`和`vehicle_j`分别装载了i，j种不同的订单（订单按照不同的卸货点分类）。对于两台无人机的任意一条运输路径`path_i`,` path_j`，定义：
   $$
   path_i = depot - drop\_point_{i1} - ...drop\_point_{ii}-depot\\
   path_j = depot-drop\_point_{j1} - ...drop\_point_{jj}-depot
   $$
   对于`path_i`,` path_j`的合并运输路径`path_{i+j}`,定义：
   $$
   path_{i+j} = depot - drop\_point_{i1} - ...drop\_point_{ii}-drop\_point_{jj} - ...drop\_point_{j1}-depot
   $$
   因为
   $$
   dist（drop\_point_{ii}, depot）+ dist（drop\_point_{jj}, depot）> dist（drop\_point_{ii}, drop\_point_{jj}）
   $$
   所以
   $$
   dist（path_i）+ dist(path_j) > dist(path_{i + j})
   $$
   <img src="C:\Users\10715\Desktop\Advanced_Algorithm_VRP\images\proof.drawio.svg" alt="proof.drawio" style="zoom:150%;" />

   因此，可以得出结论，在满足约束条件的情况下，不断地合并无人机的订单，一定可以降低一个订单处理时间间隔内无人机的总的运输里程。



#### 6.3.2、代码实现

**1、统计最小订单处理时间间隔内应处理的订单**

```python
def check_orders_due(orders: List[Order], vehicle_speed: int, dealing_window:Tuple[float, float] ) -> List[Order]:
    due_orders = []
    for order in orders:
        depot = order.destination.depot
        depot = [depot.x, depot.y]
        transport_time = calculate_distance(depot, [order.destination.x, order.destination.y]) / vehicle_speed
        if order.time_window[1] -  transport_time <= dealing_window[1]: 
            # print(f"order.time_window:{order.time_window})")
            # print(f"dealing_window:{dealing_window})")
            # print(f"Transport time: {transport_time}")
            due_orders.append(order)
    if len(due_orders) == 0:
        return None
    else:
        return due_orders
```



**2、订单分配初始化**

```python
# allocate the orders which are due to the same destination to the same vehicle
def initial_orders_to_vehicle(orders:List[Order], vehicle_id: int) -> List[Vehicle]:
    allocated_vehicles = []
    vehicle_capacity = 20
    speed = 1
    # calculate the number of vehicles needed to deliver the orders
    num_vehicles_needed = math.ceil(sum([order.demand for order in orders]) / vehicle_capacity)
    for i in range(num_vehicles_needed):
        allocated_vehicles = initialize_vehicles(num_vehicles_needed, vehicle_capacity, 1)
    
    # allocate the orders to the vehicles
    id = vehicle_id
    i = 0
    while len(orders) != 0:
        while allocated_vehicles[i].load < vehicle_capacity:
            allocated_vehicles[i].vehicle_id = id + i  + 1
            allocated_vehicles[i].orders.append(orders.pop(0))
            allocated_vehicles[i].load += 1
            if allocated_vehicles[i].load == vehicle_capacity:
                i += 1
            if len(orders) == 0:
                break
    return allocated_vehicles

# initialize the vehicle plan for the orders that are due in the current dealing window, orders's destination are the same can be delivered by the same vehicle
def initial_delivery_plan(due_orders:List[Order] , all_vehicles:List[Vehicle], depot: Tuple[float, float], current_time:float) -> List[Vehicle]:
    # classify the orders by its destination
    orders_by_destination = {}
    for order in due_orders:
        if order.destination.id in orders_by_destination:
            orders_by_destination[order.destination.id].append(order)
        else:
            orders_by_destination[order.destination.id] = [order]
    
    allocated_vehicles = []
    vehicle_id = len(all_vehicles)
    # initialize the route for each vehicle, each vechicle deals with the orders that are due at the same drop point
    for destination_id, orders in orders_by_destination.items():
        if len(orders) != 0:
            allocated_vehicles_i = initial_orders_to_vehicle(orders, vehicle_id = vehicle_id)
            vehicle_id += len(allocated_vehicles_i)
            allocated_vehicles.extend(allocated_vehicles_i)

    # generate the route for each vehicle
    for vehicle in allocated_vehicles:
        check_path(vehicle.orders, depot, vehicle, current_time, route=None)

    return allocated_vehicles
```



**3、针对未满载的无人机进行订单再分配**

```python
# aggregate vechicles until there is no more vehicle can be aggregated
def vehicles_aggregate(vehicles: List[Vehicle], depot: Tuple[float, float],current_time:float) -> List[Vehicle]:
    start_id = vehicles[0].vehicle_id 
    aggregatation_flag = True
    while aggregatation_flag:
        aggregatation_flag = False
        for i in range(len(vehicles)):
            for j in range(i+1, len(vehicles)):
                #print(f"i: {i}, j: {j}")
                if check_aggregate_capacity(vehicles[i], vehicles[j]) and check_aggregate_time(vehicles[i], vehicles[j], depot, current_time):
                    vehicles[i].orders.extend(vehicles[j].orders)
                    vehicles[i].load += vehicles[j].load
                    vehicles.pop(j)
                    aggregatation_flag = True
                    break
            if aggregatation_flag:
                break
    
    for vehicle in vehicles:
        vehicle.vehicle_id = start_id
        start_id += 1

    return vehicles
```

这些代码中还有一些检查无人机订单分配的约束满足条件以及订单时间约束满足条件的函数。这里没有附上，后面会给全部的源码。



### 6.4、车辆最短路径模型

在完成了订单的分配之后，每台无人机的订单确定。则需要在订单确定的情况下进行路径规划，以最小化无人机的运输里程。这个问题类似TSP问题，不过带有时间约束，即满足该台无人机的订单的时间需求的约束，求最短路径。考虑使用遗传算法求解，

#### 6.4.1、主要步骤 

1. **初始化种群**：初始化种群列表`route_population`。 生成初始种群，数量为`initial_population_num`。每条路线为所有订单的卸货点的随机排列。

2. **遗传算法迭代**：遗传算法的迭代次数为`generation_num`。 在每一代中执行以下操作：选择满足时间窗口约束以及无人机里程上限约束的路线`selected_routes`。 计算选择路线的适应度，即路线总距离`selected_fitness_distances`。 选择适应度最小的路线作为最优路线`optimal_route`。根据对应的适应度，使用轮盘赌模型选取父代路径进行`order crossover`以及`mutate`以生成子代路径，一些遗传算法的细节参数以及交叉互换，变异的策略细节详见代码实现。由于路径需要合法，且每个卸货点只经过一次，因此，变异策略不能使用传统的变异策略，变异只能发生再一个个体上。变异策略考虑交换随机次数的随机两个卸货点的位置。

3. **更新车辆最短路径**：每代选出最短的路径，经过一定的遗传代数后，选出整个过程中最短的路径作为该台无人机的最短路径。

对一个订单处理时间间隔内，对已经完成订单分配的所有无人机使用遗传算法搜索满足约束的最短路径。



#### 6.4.2、代码实现

```python
#using the GA to find the shortest path for the vehicle to deliver the orders
def find_shortest_path_GA(vehicle:Vehicle, depot: Tuple[float, float], current_time:float) -> Vehicle:
    #initialize the population
    route_populations = []

    initial_population_num = 5
    generation_num = 5
    order_crossover_rate = 0.8
    mutation_rate = 0.3
    
    original_route = vehicle.route.copy()
    if len(original_route) <5:
        generation_num = 1
    # generate the initial population
    for i in range(initial_population_num):
        route_population = []
        original_route = vehicle.route.copy()
        while (len(original_route)):
            point = random.choice(original_route)
            route_population.append(point)
            original_route.remove(point)
        route_populations.append(route_population)
            
    route_populations.append(vehicle.route)
    
    optimal_route = vehicle.route.copy()
    for generation in range(generation_num):
        #print(f"Generation: {generation}")

        #select the route which satisfy the i time window constraints
        selected_routes = []
        for i in range(len(route_populations)):
            if check_path(vehicle.orders, depot, vehicle, start_time=current_time, route=route_populations[i]):
                selected_routes.append(route_populations[i])
        
        if selected_routes == []:
            #print("No route satisfy the time window constraints.")
            break

        #calculate the fitness of the selected routes
        selected_fitness_distances = []
        for route in selected_routes:
            fitness_distance = calculate_path_mileage(route, depot)
            selected_fitness_distances.append(fitness_distance)
       
        #select the shortest route as the optimal route
        i_optimal_route = selected_routes[selected_fitness_distances.index(min(selected_fitness_distances))]
    
        #update the optimal route
        if min(selected_fitness_distances) < calculate_path_mileage(optimal_route, depot):
            optimal_route = i_optimal_route
            vehicle.route = optimal_route
            vehicle.real_mileage = calculate_path_mileage(optimal_route, depot)

        if generation_num >  1:
            #calculate the probability of each route to be selected as the parent route
            fitness_sum = sum(1/selected_fitness_distance for selected_fitness_distance in selected_fitness_distances)
            probabilities = [fitness_distance/fitness_sum for fitness_distance in selected_fitness_distances]

            #using the roulette wheel selection to select the parent routes
            selected_routes = random.choices(selected_routes, weights=probabilities, k=initial_population_num)
            
            next_generation = selected_routes

            for i in range(len(selected_routes)):
                # generate a probability between 0 and 1, if the probability is less than the order crossover rate, the route will be crossovered
                if random.random() < order_crossover_rate:
                    #select another route to crossover
                    route2 = random.choice(selected_routes)
                    route1 = selected_routes[i]
                    [route1, route2] = order_crossover(route1, route2)
                    next_generation.append(route1)
                    next_generation.append(route2)

                #generate a probability between 0 and 1, if the probability is less than the mutation rate, the route will be mutated
                if random.random() < mutation_rate:
                    mutated_route = mutation(selected_routes[i])
                    next_generation.append(mutated_route)

            route_population = next_generation
    
    #calculate the total distance of the optimal route
    return vehicle

def mutation(selected_route:List[DropPoint]) -> List[DropPoint]:
    num_mutated_points = random.randint(1, len(selected_route))
    # shuffle the mutated points in the selected route
    for i in range(num_mutated_points):
        k = random.randint(0, len(selected_route)-1)
        j = random.randint(0, len(selected_route)-1)
        if k != j:
            selected_route[k], selected_route[j] = selected_route[j], selected_route[k]
    return selected_route

def order_crossover(route1:List[DropPoint], route2:List[DropPoint]) -> Tuple[List[DropPoint], List[DropPoint]]:
    #select two points randomly from the route
    point1 = random.randint(0, len(route1)-1)
    point2 = random.randint(0, len(route1)-1)
    if point1 > point2:
        point1, point2 = point2, point1
    #find crossover part
    crossover_part1 = route1[point1:point2]
    crossover_part2 = route2[point1:point2]

    left_route1 = []
    left_route2 = []

    for i in route1:
        if i not in crossover_part2:
            left_route1.append(i)
    
    for j in route2:
        if j not in crossover_part1:
            left_route2.append(j)
    
    #insert the crossover part2 to the route1
    new_route1 = route1
    new_route1[:point1] = left_route1[:point1]
    new_route1[point2:] = left_route1[point1:]
    new_route1[point1:point2] = crossover_part2

    #insert the crossover part1 to the route2
    new_route2 = route2
    new_route2[:point1] = left_route2[:point1]
    new_route2[point2:] = left_route2[point1:]
    new_route2[point1:point2] = crossover_part1
   
    return [new_route1, new_route2]
```



## 7、模拟实验

模拟实验主要是测试订单处理时间间隔的大小(即m值)对最终总的无人机的运输距离的影响。

模拟的一些基本参数如下：

```python
    num_drop_points = 40 # number of drop points
    num_depots = 5 # number of depots
    vehicle_speed = 1  # 1 unit distance per minute
    max_distance = 5 # max distance of the map a circle with radius 5
    
    time_interval = 12  # interval of generating orders
    time_sensitivity = 1 # minute unit in the simulation
    simulation_duration = 8 * 60  # simulation duration in minutes
    max_orders = 5  # Max orders generated per generating interval per drop point

```



m值的取值范围设定在[1,10]之间，每组实验固定m值并进行20次模拟取平均值作为该组实验的最终的全局无人机实际里程总和。

将所有代码文件放在一个文件夹下：

```
--VRP_problem
------utils.py
------drop_point_classify.py
------GA_shortest_path.py
------orders_processing.py
------vrp_simulator.py
------main.py
```

执行main.py文件即可开始模拟实验，模拟实验的结果如下：

<img src="C:\Users\10715\AppData\Roaming\Typora\typora-user-images\image-20240626154019137.png" alt="image-20240626154019137" style="zoom:67%;" />

可以当前设置的参数情况下，设置`dealing_window = 2 * t`，总的无人机的里程最小。当然，也可以调整其他参数进行模拟。





## 8、附件

### 8.1、完整代码

#### 8.1.1、utils.py

```python
from typing import List, Tuple
import random
import math

class Vehicle:
    def __init__(self, vehicle_id: int, capacity: int, speed: int ):
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.route = []
        self.load = 0
        self.speed = speed  # 1 unit distance per minute
        self.orders = []
        self.real_mileage = 0
        self.max_mileage = 20
        
class DropPoint:
    def __init__(self, x: float, y: float,id:int):
        self.x = x
        self.y = y
        self.id = id
        self.depot = None

class Order:
    def __init__(self, order_id: int,  destination: DropPoint, demand: int, time_window: Tuple[int, int], priority: int):
        self.order_id = order_id
        self.destination = destination
        self.demand = demand
        self.time_window = time_window
        self.priority = priority
        

class DePot:
    def __init__(self, x: float, y: float, id:int):
        self.x = x
        self.y = y
        self.id =id
        


def initialize_vehicles(num_vehicles: int, capacity: int, speed: int) -> List[Vehicle]:
    vehicles = []
    for i in range(num_vehicles):
        vehicles.append(Vehicle(i, capacity, speed))
    return vehicles

def initialize_drop_points(num_drop_points: int,max_distance:float) -> List[DropPoint]:
    drop_points = []
    thetas = [random.uniform(0, 2 * math.pi) for _ in range(num_drop_points)]
    distances = [random.uniform(0, max_distance) for _ in range(num_drop_points)]

    drop_point = [(distances[i]* math.cos(thetas[i]), distances[i] * math.sin(thetas[i])) for i in range(len(thetas))]
    for k in range(num_drop_points):
        x = drop_point[k][0]
        y = drop_point[k][1]
        drop_points.append(DropPoint(x, y, k + 1))
    return drop_points

def initialize_depots(num_depots, radium: float) -> List[DePot]:
    thetas = [random.uniform(0, 2 * math.pi) for _ in range(num_depots)]
    depots = []
    depot = [(radium* math.cos(thetas[i]), radium * math.sin(thetas[i])) for i in range(len(thetas))]
    for k in range(num_depots):
        x = depot[k][0]
        y = depot[k][1]
        depots.append(DePot(x, y, k+1))
    return depots

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# calculate the distance the vehicle needs to travel to deliver the orders
def calculate_route_distance(vehicle: Vehicle, depot: Tuple[float, float]) -> float:
    route_distance = 0
    for i in range(len(vehicle.orders)):
        if i == 0:
            route_distance += calculate_distance(depot, [vehicle.orders[i].destination.x, vehicle.orders[i].destination.y])
        else:
            route_distance += calculate_distance([vehicle.orders[i-1].destination.x, vehicle.orders[i-1].destination.y], [vehicle.orders[i].destination.x, vehicle.orders[i].destination.y])
    route_distance += calculate_distance([vehicle.orders[-1].destination.x, vehicle.orders[-1].destination.y], depot)
    return route_distance
```

#### 8.1.2、drop_point_classify.py

```python
from typing import List

from utils import DropPoint, DePot, calculate_distance

#using the KNN algorithm to find the depot that is closest to the drop point
def find_closest_depot(drop_points: List[DropPoint], depots: List[DePot], therehold: float):
    for drop_point in drop_points:
        # 统计当前drop point半径therehold范围内的drop point的对应的不同depot的数量
        depot_count = {}
        for drop_point_ in drop_points:
            if drop_point_ != drop_point:
                distance = calculate_distance([drop_point.x, drop_point.y], [drop_point_.x, drop_point_.y])
                if distance <= therehold and drop_point_.depot is not None:
                    if drop_point_.depot.id in depot_count:
                        depot_count[drop_point_.depot.id] += 1
                    else:
                        depot_count[drop_point_.depot.id] = 1

        # 找到当前drop point半径therehold范围内的drop point的对应的不同depot的数量最多的depot,并将当前drop point的depot设置为这个depot
        if len(depot_count) != 0:
            max_count = max(depot_count.values())
            for depot_id, count in depot_count.items():
                if count == max_count:
                    drop_point.depot = depots[depot_id-1]
                    #print(f"Drop point {drop_point.id} is closest to depot {depot_id}.")
                    break
        else:
            # 如果当前drop point半径therehold范围内没有drop point对应的depot,则将当前drop point的depot设置为距离当前drop point最近的depot
            min_distance = float('inf')
            for depot in depots:
                distance = calculate_distance([drop_point.x, drop_point.y], [depot.x, depot.y])
                if distance < min_distance:
                    min_distance = distance
                    drop_point.depot = depot
            #print(f"Drop point {drop_point.id} is closest to depot {drop_point.depot.id}.")

```

#### 8.1.3、orders_processing.py

```python
from typing import List, Tuple
import random
import math
from utils import calculate_distance, Order, Vehicle, DropPoint, initialize_vehicles

# classify the orders by its destination belonging to the sam depot
def classify_orders_by_depot(orders: List[Order]) -> List[Order]:
    orders_by_depot = {}
    for order in orders:
        if order.destination.depot.id in orders_by_depot:
            orders_by_depot[order.destination.depot.id].append(order)
        else:
            orders_by_depot[order.destination.depot.id] = [order]
    return orders_by_depot

def generate_orders(current_orders_num: int, drop_points: List[DropPoint], current_time: int, max_orders: int) -> List[Order]:
    orders = []
    order_id = current_orders_num
    for i in range(len(drop_points)):
        for j in range(random.randint(0, max_orders)):
            order_id += 1
            destination = drop_points[i]
            demand = 1  # Assuming each order has a demand of 1
            priority = random.choice([1, 2, 3])
            if priority == 1:
                time_window = (current_time, current_time + 180)  # 3 hours
            elif priority == 2:
                time_window = (current_time, current_time + 90)   # 1.5 hours
            else:
                time_window = (current_time, current_time + 30)   # 0.5 hour
            #print(order_id, destination,destination.id, demand, time_window, priority)
            orders.append(Order(order_id,destination , demand, time_window, priority))
    return orders

def sort_orders_by_window_end(orders: List[Order]) -> List[Order]:
    return sorted(orders, key=lambda order: order.time_window[1])

def check_orders_due(orders: List[Order], vehicle_speed: int, dealing_window:Tuple[float, float] ) -> List[Order]:
    due_orders = []
    for order in orders:
        depot = order.destination.depot
        depot = [depot.x, depot.y]
        transport_time = calculate_distance(depot, [order.destination.x, order.destination.y]) / vehicle_speed
        if order.time_window[1] -  transport_time <= dealing_window[1]: 
            # print(f"order.time_window:{order.time_window})")
            # print(f"dealing_window:{dealing_window})")
            # print(f"Transport time: {transport_time}")
            due_orders.append(order)
    if len(due_orders) == 0:
        return None
    else:
        # print(len(due_orders))
        # for due_order in due_orders:
        #     print(f"Due orders at drop point {due_order.destination.id}:")
        #     print(f"Order ID: {due_order.order_id}, destination.id: {due_order.destination.id},depot_id:{due_order.destination.depot.id} Demand: {due_order.demand}, Time Window: {due_order.time_window}, Priority: {due_order.priority}")
        return due_orders

# remove the orders that are due from the orders_to_be_delivered list
def remove_due_orders(orders_to_be_delivered: List[Order], due_orders:List[Order]) -> List[Order]:
    for due_order in due_orders:
        orders_to_be_delivered.remove(due_order)
    return orders_to_be_delivered

# allocate the orders which are due to the same destination to the same vehicle
def initial_orders_to_vehicle(orders:List[Order], vehicle_id: int) -> List[Vehicle]:
    allocated_vehicles = []
    vehicle_capacity = 30

    # calculate the number of vehicles needed to deliver the orders
    num_vehicles_needed = math.ceil(sum([order.demand for order in orders]) / vehicle_capacity)
    
    for i in range(num_vehicles_needed):
        allocated_vehicles = initialize_vehicles(num_vehicles_needed, vehicle_capacity, 1)
    
    # allocate the orders to the vehicles
    id = vehicle_id
    i = 0
    while len(orders) != 0:
        while allocated_vehicles[i].load < vehicle_capacity:
            allocated_vehicles[i].vehicle_id = id + i  + 1
            allocated_vehicles[i].orders.append(orders.pop(0))
            allocated_vehicles[i].load += 1
            if allocated_vehicles[i].load == vehicle_capacity:
                i += 1
            if len(orders) == 0:
                # print("All orders are allocated to vehicles.")
                break
    return allocated_vehicles

# initialize the vehicle plan for the orders that are due in the current dealing window, orders's destination are the same can be delivered by the same vehicle
def initial_delivery_plan(due_orders:List[Order] , all_vehicles:List[Vehicle], depot: Tuple[float, float], current_time:float) -> List[Vehicle]:
    # classify the orders by its destination
    orders_by_destination = {}
    for order in due_orders:
        if order.destination.id in orders_by_destination:
            orders_by_destination[order.destination.id].append(order)
        else:
            orders_by_destination[order.destination.id] = [order]
    
    allocated_vehicles = []
    vehicle_id = len(all_vehicles)
    # initialize the route for each vehicle, each vechicle deals with the orders that are due at the same drop point
    for destination_id, orders in orders_by_destination.items():
        if len(orders) != 0:
            allocated_vehicles_i = initial_orders_to_vehicle(orders, vehicle_id = vehicle_id)
            vehicle_id += len(allocated_vehicles_i)
            # print(f"Number of vehicles allocated: {len(allocated_vehicles_i)}")
            allocated_vehicles.extend(allocated_vehicles_i)

    # generate the route for each vehicle
    for vehicle in allocated_vehicles:
        check_path(vehicle.orders, depot, vehicle, current_time, route=None)

    return allocated_vehicles

# aggregate vechicles until there is no more vehicle can be aggregated
def vehicles_aggregate(vehicles: List[Vehicle], depot: Tuple[float, float],current_time:float) -> List[Vehicle]:
    start_id = vehicles[0].vehicle_id 
    aggregatation_flag = True
    while aggregatation_flag:
        aggregatation_flag = False
        for i in range(len(vehicles)):
            for j in range(i+1, len(vehicles)):
                # print(f"aggregate vehicle {vehicles[i].vehicle_id} and vehicle {vehicles[j].vehicle_id}")
                if check_aggregate_capacity(vehicles[i], vehicles[j]) and check_aggregate_time(vehicles[i], vehicles[j], depot, current_time):
                    vehicles[i].orders.extend(vehicles[j].orders)
                    vehicles[i].load += vehicles[j].load
                    vehicles.pop(j)
                    aggregatation_flag = True
                    break
            if aggregatation_flag:
                break
    
    for vehicle in vehicles:
        vehicle.vehicle_id = start_id
        start_id += 1

    return vehicles
                
# check the capacity of the aggregated vehicle
def check_aggregate_capacity(vehicle1: Vehicle, vehicle2: Vehicle) -> bool:
    capacity = vehicle1.capacity
    if vehicle1.load + vehicle2.load <= capacity:
        #print("The aggregated vehicle has enough capacity.")
        return True
    else:
        #print("The aggregated vehicle does not have enough capacity.")
        return False

# check the time is enough for the aggregated vehicle to deliver the orders
def check_aggregate_time(vehicle1: Vehicle, vehicle2: Vehicle, depot: Tuple[float, float], start_time:float) -> bool:
    orders = []
    orders.extend(vehicle1.orders)
    orders.extend(vehicle2.orders)
    if check_path(orders, depot, vehicle1,start_time, route=None):
        #print("The aggregated vehicle can deliver the orders in time.")
        return True
    else:
        #print("The aggregated vehicle cannot deliver the orders in time.")
        return False


def generate_path_orders_from_orders(orders: List[Order], speed: float,route:List[DropPoint] ) -> List[Order]:
    path_orders = []
    speed = speed
        
    #classify the orders by its destination
    orders_by_destination = {}
    for order in orders:
        if order.destination.id in orders_by_destination:
            orders_by_destination[order.destination.id].append(order)
        else:
            orders_by_destination[order.destination.id] = [order]
    
    # find the orders that are due the earliest at each drop point
    for destination_id, orders in orders_by_destination.items():
        orders = sort_orders_by_window_end(orders)
        path_orders.append(orders[0])
    
    sorted_path_orders = []
    if route is not None:
        for i in range(len(route)):
            destination_id = route[i].id
            sorted_path_order = [order for order in path_orders if order.destination.id == destination_id]
            sorted_path_orders.extend(sorted_path_order)
    else:
        sorted_path_orders = path_orders
    
    return sorted_path_orders


#check the path is available for the vehicle to deliver the orders
def check_path(vehicle_orders: List[Order], depot: Tuple[float, float], vehicle:Vehicle, start_time:float, route:List[DropPoint]) -> bool:
    vehicle_orders = sort_orders_by_window_end(vehicle_orders)
    speed = vehicle.speed
    path_orders = generate_path_orders_from_orders(vehicle_orders, speed, route)
    real_window_ends = []
    # calculate the real window end for each order
    current_time = start_time
    for i in range(len(path_orders)):
        if i == 0:
            transport_time = calculate_distance(depot, [path_orders[i].destination.x, path_orders[i].destination.y]) / speed
            real_window_end = current_time + transport_time
            real_window_ends.append(real_window_end)
            current_time = real_window_end
        else:
            transport_time = calculate_distance([path_orders[i-1].destination.x, path_orders[i-1].destination.y], [path_orders[i].destination.x, path_orders[i].destination.y]) / speed
            real_window_end = current_time + transport_time
            real_window_ends.append(real_window_end)
            current_time = real_window_end

    for i in range(len(path_orders)):
        if real_window_ends[i] > path_orders[i].time_window[1]:
            return False
    
    vehicle_route = []
    for path_order in path_orders:
        vehicle_route.append(path_order.destination)
    real_mileage = calculate_path_mileage(vehicle_route, depot)
    if  real_mileage <= vehicle.max_mileage:
        vehicle.real_mileage = real_mileage
        vehicle.route = vehicle_route
        return True
    else:
        #print("out of the max mileage")
        return False

def calculate_path_mileage(vehicle_path:List[DropPoint], depot: Tuple[float, float]) -> float:
    vehicle_real_mileage = 0
    for i in range(len(vehicle_path)):
        if i == 0:
            vehicle_real_mileage += calculate_distance(depot, [vehicle_path[i].x, vehicle_path[i].y])
        else:
            vehicle_real_mileage += calculate_distance([vehicle_path[i-1].x,vehicle_path[i-1].y], [vehicle_path[i].x, vehicle_path[i].y])
    vehicle_real_mileage += calculate_distance([vehicle_path[-1].x,vehicle_path[-1].y], depot)
    return vehicle_real_mileage
```

#### 8.1.4、GA_shortest_path.py

```python
from typing import List, Tuple
import random
import math
from utils import Vehicle, DropPoint, calculate_distance
from orders_processing import check_path,calculate_path_mileage
import sys

#using the GA to find the shortest path for the vehicle to deliver the orders
def find_shortest_path_GA(vehicle:Vehicle, depot: Tuple[float, float], current_time:float) -> Vehicle:
    #initialize the population
    route_populations = []

    initial_population_num = 5
    generation_num = 5
    order_crossover_rate = 0.8
    mutation_rate = 0.3
    
    original_route = vehicle.route.copy()
    if len(original_route) <5:
        generation_num = 1
    # generate the initial population
    for i in range(initial_population_num):
        route_population = []
        original_route = vehicle.route.copy()
        while (len(original_route)):
            point = random.choice(original_route)
            route_population.append(point)
            original_route.remove(point)
        route_populations.append(route_population)
            
    route_populations.append(vehicle.route)
    
    optimal_route = vehicle.route.copy()
    for generation in range(generation_num):
        #print(f"Generation: {generation}")

        #select the route which satisfy the i time window constraints
        selected_routes = []
        for i in range(len(route_populations)):
            if check_path(vehicle.orders, depot, vehicle, start_time=current_time, route=route_populations[i]):
                selected_routes.append(route_populations[i])
        
        if selected_routes == []:
            #print("No route satisfy the time window constraints.")
            break

        #calculate the fitness of the selected routes
        selected_fitness_distances = []
        for route in selected_routes:
            fitness_distance = calculate_path_mileage(route, depot)
            selected_fitness_distances.append(fitness_distance)
        

        #select the shortest route as the optimal route
        i_optimal_route = selected_routes[selected_fitness_distances.index(min(selected_fitness_distances))]
    
        #update the optimal route
        if min(selected_fitness_distances) < calculate_path_mileage(optimal_route, depot):
            optimal_route = i_optimal_route
            vehicle.route = optimal_route
            vehicle.real_mileage = calculate_path_mileage(optimal_route, depot)


        if generation_num >  1:
            #calculate the probability of each route to be selected as the parent route
            fitness_sum = sum(1/selected_fitness_distance for selected_fitness_distance in selected_fitness_distances)
            probabilities = [fitness_distance/fitness_sum for fitness_distance in selected_fitness_distances]

            #using the roulette wheel selection to select the parent routes
            selected_routes = random.choices(selected_routes, weights=probabilities, k=initial_population_num)
            
            next_generation = selected_routes

            for i in range(len(selected_routes)):
                # generate a probability between 0 and 1, if the probability is less than the order crossover rate, the route will be crossovered
                if random.random() < order_crossover_rate:
                    #select another route to crossover
                    route2 = random.choice(selected_routes)
                    route1 = selected_routes[i]
                    [route1, route2] = order_crossover(route1, route2)
                    next_generation.append(route1)
                    next_generation.append(route2)

                #generate a probability between 0 and 1, if the probability is less than the mutation rate, the route will be mutated
                if random.random() < mutation_rate:
                    mutated_route = mutation(selected_routes[i])
                    next_generation.append(mutated_route)

            route_population = next_generation
    
    #calculate the total distance of the optimal route
    return vehicle

def mutation(selected_route:List[DropPoint]) -> List[DropPoint]:
    num_mutated_points = random.randint(1, len(selected_route))
    # shuffle the mutated points in the selected route
    for i in range(num_mutated_points):
        k = random.randint(0, len(selected_route)-1)
        j = random.randint(0, len(selected_route)-1)
        if k != j:
            selected_route[k], selected_route[j] = selected_route[j], selected_route[k]
    return selected_route

def order_crossover(route1:List[DropPoint], route2:List[DropPoint]) -> Tuple[List[DropPoint], List[DropPoint]]:
    #select two points randomly from the route
    point1 = random.randint(0, len(route1)-1)
    point2 = random.randint(0, len(route1)-1)
    if point1 > point2:
        point1, point2 = point2, point1
    #find crossover part
    crossover_part1 = route1[point1:point2]
    crossover_part2 = route2[point1:point2]

    left_route1 = []
    left_route2 = []

    for i in route1:
        if i not in crossover_part2:
            left_route1.append(i)
    
    for j in route2:
        if j not in crossover_part1:
            left_route2.append(j)
    
    #insert the crossover part2 to the route1
    new_route1 = route1
    new_route1[:point1] = left_route1[:point1]
    new_route1[point2:] = left_route1[point1:]
    new_route1[point1:point2] = crossover_part2

    #insert the crossover part1 to the route2
    new_route2 = route2
    new_route2[:point1] = left_route2[:point1]
    new_route2[point2:] = left_route2[point1:]
    new_route2[point1:point2] = crossover_part1
   
    return [new_route1, new_route2]
    
```



#### 7.1.5、vrp_simulator.py

```python
import random
import math
from typing import List, Tuple
import json
from utils import initialize_drop_points, initialize_depots
from drop_point_classify import find_closest_depot
from GA_shortest_path import find_shortest_path_GA
from orders_processing import generate_orders, sort_orders_by_window_end, check_orders_due, classify_orders_by_depot, initial_delivery_plan, vehicles_aggregate, remove_due_orders

#simulate the whole process
def simulator(k:int)-> float:
    num_drop_points = 40 # number of drop points
    num_depots = 5 # number of depots
    vehicle_speed = 1  # 1 unit distance per minute
    max_distance = 5 # max distance of the map a circle with radius 5
    
    time_interval = 12  # interval of generating orders
    time_sensitivity = 1 # minute unit in the simulation
    simulation_duration = 8 * 60  # simulation duration in minutes
    max_orders = 5  # Max orders generated per generating interval per drop point

    #generate drop points which distances from depot limited to max_distance randomly
    drop_points = initialize_drop_points(num_drop_points,max_distance)
    
    #generate depots which distances from the origin limited to max_distance randomly
    depots = initialize_depots(num_depots, max_distance)

    current_time = 0

    orders_to_be_delivered = []

    all_vehicles = []
    current_orders_num = 0
    dealing_window = [0,0]

    #find the closest depot for each drop point
    find_closest_depot(drop_points, depots, therehold=0.5)

    while current_time < simulation_duration:
        if current_time % time_interval == 0:
            new_orders = generate_orders(current_orders_num,drop_points, current_time, max_orders)
            current_orders_num += len(new_orders)
            orders_to_be_delivered.extend(new_orders)
            orders_to_be_delivered = sort_orders_by_window_end(orders_to_be_delivered)

            # set the dealing window to be the next k time intervals
            dealing_window = [current_time, current_time + k * time_interval]
           
            #check orders due in the current dealing  window
            due_orders = check_orders_due(orders_to_be_delivered, vehicle_speed=vehicle_speed, dealing_window=dealing_window)

            if not due_orders:
                None
                # print("No orders due in the current dealing window.")
            else:
                due_orders_by_depots = classify_orders_by_depot(due_orders)
                for depot_id, due_orders_by_depot in due_orders_by_depots.items():
                    # print(f'dealing the orders starting from depot {depot_id}...')
                    depot = depots[depot_id-1]
                    depot = [depot.x, depot.y]
                    #initialize the vehicle plan for the orders that are due in the current dealing window, orders's destination are the same can be delivered by the same vehicle
                    allocated_vehicles = initial_delivery_plan(due_orders_by_depot, all_vehicles, depot, current_time)

                    #aggregate vechicles until there is no more vehicle can be aggregated
                    aggregated_vehicles = vehicles_aggregate(allocated_vehicles, depot, current_time)

                    for aggregated_vehicle in aggregated_vehicles:
                        
                        # if check_path(aggregated_vehicle.orders, depot, aggregated_vehicle, start_time=current_time, route=aggregated_vehicle.route) == False:
                        #     print("error: the aggregated vehicle's route is not valid")
                        #     sys.exit(1)
                        
                        # print(f"real mileage of vehicle {aggregated_vehicle.vehicle_id}: {aggregated_vehicle.real_mileage}")
                        # #using the GA to find the shortest path for the vehicle to deliver the orders
                        # print(f"searching for the shortest path for vehicle {aggregated_vehicle.vehicle_id}")
                        aggregated_vehicle = find_shortest_path_GA(aggregated_vehicle, depot, current_time= current_time)


                    #print the aggregated vehicles' orders and routes
                    # for aggregated_vehicle in aggregated_vehicles:
                    #     print(f"Aggregated vehicle {aggregated_vehicle.vehicle_id} has orders:")
                    #     for order in aggregated_vehicle.orders:
                    #         print(f"Order ID: {order.order_id}, destination.id: {order.destination.id}, Demand: {order.demand}, Time Window: {order.time_window}, Priority: {order.priority}")
                    #     print(f"Aggregated vehicle {aggregated_vehicle.vehicle_id} has route:")
                    #     for drop_point in aggregated_vehicle.route:
                    #         print(f"Drop point ID: {drop_point.id}")

                    all_vehicles.extend(aggregated_vehicles)

                    #remove the orders that are due from the orders_to_be_delivered list
                    orders_to_be_delivered = remove_due_orders(orders_to_be_delivered, due_orders_by_depot)
            
        current_time += time_sensitivity
    
    #write all the vehicles' orders and routes to a json file
    whole_vehicle_distance = 0
    vehicles = []
    error_count = 0
    for vehicle in all_vehicles:
        vehicle_dict = {}
        vehicle_dict['vehicle_id'] = vehicle.vehicle_id
        vehicle_dict['orders'] = []
        for order in vehicle.orders:
            order_dict = {}
            order_dict['order_id'] = order.order_id
            order_dict['destination_id'] = order.destination.id
            order_dict['demand'] = order.demand
            order_dict['time_window'] = order.time_window
            order_dict['priority'] = order.priority
            vehicle_dict['orders'].append(order_dict)
        vehicle_dict['route'] = [drop_point.id for drop_point in vehicle.route]
        vehicle_dict['real_mileage'] = vehicle.real_mileage
        if vehicle.real_mileage > vehicle.max_mileage:
            error_count += 1
        whole_vehicle_distance += vehicle.real_mileage
        vehicles.append(vehicle_dict)

    with open('vehicles.json', 'w') as f:
        json.dump(vehicles, f, indent=4)
    
    print(f"Whole vehicles' distance: {whole_vehicle_distance}")
    return whole_vehicle_distance



```



#### 7.1.6、main.py

```python
from vrp_simulator import simulator
import json
import matplotlib.pyplot as plt

def main():
    #run the simulator with different k values and save the results into a json file
    #set the m as the reconducting times
    k = 10
    n = 20
    results = {}
    for i in range (1, k + 1):
        results[str(i)] = []
        for j in range(n):
            print(f"Simulating with k = {i} for the {j+1}th time...")
            results[str(i)].append(simulator(k=i))
    
    with open('results.json', 'w') as f:
        json.dump(results, f)
    
    average_results = {}
    for i in range(1, k + 1):
        average_results[str(i)] = sum(results[str(i)])/len(results[str(i)])

    #plot the results and save the plot into a pdf file
    plt.plot(average_results.keys(), average_results.values())
    plt.xlabel('k')
    plt.ylabel('Whole vehicles distance')
    plt.title('Whole vehicles distance vs. k')
    plt.savefig('whole_vehicles_distance_vs_k.pdf')


if __name__ == "__main__":
    main()
```



### 8.2、代码运行说明及结果

将所有代码文件放在一个文件夹下：

```
--VRP_problem
------utils.py
------drop_point_classify.py
------GA_shortest_path.py
------orders_processing.py
------vrp_simulator.py
------main.py
```



执行以下命令即可模拟订单生成并执行上述模型的操作

```shell
python3 main.py
```

运行结果是所有调度过的无人机，以及这些无人机装载的订单，最终运输订单的路径。

保存在vehicles.json文件中,例子如下：

```json
[
    {
        "vehicle_id": 1,
        "orders": [
            {
                "order_id": 1,
                "destination_id": 2,
                "demand": 1,
                "time_window": [
                    0,
                    30
                ],
                "priority": 3
            },
            {
                "order_id": 10,
                "destination_id": 6,
                "demand": 1,
                "time_window": [
                    0,
                    30
                ],
                "priority": 3
            },
            {
                "order_id": 12,
                "destination_id": 6,
                "demand": 1,
                "time_window": [
                    0,
                    30
                ],
                "priority": 3
            },
            {
                "order_id": 13,
                "destination_id": 6,
                "demand": 1,
                "time_window": [
                    0,
                    30
                ],
                "priority": 3
            }
        ],
        "route": [
            6,
            2
        ]
    },
```

