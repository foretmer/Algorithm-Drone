import csv
from datetime import datetime, timedelta
from const import (
    MAX_NON_URGENT_ORDERS_PER_PERIOD,
    NUM_DC,
    NUM_DP,
    DURATION_TIME,
    MAX_CAPACITY,
    START_TIME,
    UNIT_TIME,
)


class Solver:
    def __init__(self, order_file_path, dist_file_path):
        self.orders = []
        self.order_group = {}
        self.dist_matrix = [
            [None for _ in range(NUM_DP + NUM_DC)] for _ in range(NUM_DP + NUM_DC)
        ]
        self.current_orders = []
        self.max_non_urgent_orders_per_period = MAX_NON_URGENT_ORDERS_PER_PERIOD
        self.priority_map = {
            "紧急": 1,  # 数字越小，优先级越高
            "较紧急": 2,
            "一般": 3,
        }
        self.routes = []
        self.current_time = datetime.strptime(START_TIME, "%Y-%m-%d %H:%M:%S")
        self.__read_dist_file(dist_file_path)
        self.__read_order_file(order_file_path)

    def __increment_time(self):
        self.current_time += timedelta(minutes=UNIT_TIME)  # 每30分钟生成一次订单

    def __read_order_file(self, filename):
        # 读取CSV文件
        with open(filename, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳过标题行
            for row in csv_reader:
                order = {
                    "Order Id": row[0],
                    "Delivery Point": row[1],
                    "Priority": self.priority_map[row[2]],
                    "Deadline": datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S"),
                    "Created At": datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S"),
                }
                self.orders.append(order)
        # 根据创建时间分组订单
        for order in self.orders:
            created_at = order["Created At"]
            if created_at not in self.order_group:
                self.order_group[created_at] = []
            self.order_group[created_at].append(order)
        # print(len(self.order_group))
        # print(self.order_group.keys())

    def __read_dist_file(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳过标题行
            for row in csv_reader:
                from_point = int(row[0])
                to_point = int(row[1])
                distance = float(row[2])
                # 确保配送中心和卸货点的编号在1到15之间
                if (
                    1 <= from_point <= NUM_DP
                    and NUM_DC + 1 <= to_point <= NUM_DC + NUM_DP + 1
                ):
                    # 将距离数据存储到二维数组中
                    self.dist_matrix[from_point - 1][to_point - 1] = distance
                    self.dist_matrix[to_point - 1][from_point - 1] = distance

    def calculate_distance(self, point1, point2):
        return self.dist_matrix[point1][point2]

    def get_orders(self):
        """
        读取下一个时间内的订单
        """
        if self.current_time in self.order_group:
            self.current_orders.extend(self.order_group[self.current_time])

    def sort_orders_by_priority(self):
        """
        对当前的订单优先级进行排序
        """
        self.current_orders.sort(key=lambda x: x.get("priority", float("inf")))

    def plan_route(self):
        """
        规划无人机的配送路线，目标是最小化总配送路径长度。
        """
        routes = []
        routes.extend(self.handle_urgent_orders())
        routes.extend(self.handle_non_urgent_orders())
        routes = self.aggregate_routes(routes)
        print(routes)
        self.routes.append(routes)

    def handle_urgent_orders(self):
        urgent_orders = [
            order for order in self.current_orders if order["Priority"] == 1
        ]
        if not urgent_orders:
            return

        # 为每个紧急订单分配最近的配送中心
        routes = []
        for order in urgent_orders:
            route = self.assign_route_to_orders(order)
            routes.append(route)
            # 从当前订单列表中移除已处理的订单
            self.current_orders.remove(order)
        return routes

    def find_dp_index(self, delivery_point):
        return int(delivery_point[2:]) - 1

    def assign_route_to_orders(self, order):
        closest_dc = self.find_closest_distribution_center(order["Delivery Point"])
        route = self.create_route_for_order(closest_dc, order)
        return route

    def calculate_route_distance(self, route_str):
        """
        根据route_str和距离矩阵计算总距离。
        """
        total_distance = 0
        points = route_str.split("-")
        for i in range(1, len(points)):
            from_point = int(points[i - 1])
            to_point = int(points[i])
            total_distance += self.dist_matrix[from_point - 1][to_point - 1]
        return total_distance * 2

    def aggregate_routes(self, routes):
        """
        聚合路线，尽量合并起始点相同的路线以减少总配送距离。
        """
        # 按路线的起始配送中心和总距离对路线进行排序，优先合并距离较短的路线
        routes.sort(key=lambda x: (x["distribution_center"], x["distance"]))

        def format_route(route):
            # 使用集合来存储已经添加的配送点索引，避免重复
            added_points = set()
            unique_points = []

            for point in route:
                if point[1] == "end" and point[0] not in added_points:
                    added_points.add(point[0])
                    unique_points.append(str(point[0]))

            # 将去重后的配送点索引连接成字符串，使用"-"作为分隔符
            return "-".join(unique_points)

        # 创建一个字典来保存合并后的路线，键为配送中心索引，值为路线列表
        aggregated_routes = {}
        for route in routes:
            dc = route["distribution_center"]
            if dc not in aggregated_routes:
                aggregated_routes[dc] = []
            aggregated_routes[dc].append(route)

        # 遍历聚合字典，合并同一配送中心的路线
        for dc, routes in aggregated_routes.items():
            if len(routes) > 1:
                # 尝试合并同一配送中心的路线
                merged_route = routes[0]
                for i in range(1, len(routes)):
                    current_route = routes[i]
                    if self.can_merge_routes(merged_route, current_route):
                        last_dp = (
                            merged_route["route"][-1][0]
                            if merged_route["route"]
                            else None
                        )
                        if last_dp:
                            merged_route["route"].append(
                                (current_route["route"][0][0], "end")
                            )
                            merged_route["orders"].extend(current_route["orders"])
                            merged_route["route"].extend(current_route["route"][1:])
                        else:
                            merged_route["route"] = current_route["route"]
                        merged_route["route_str"] = format_route(merged_route["route"])
                        merged_route["distance"] = self.calculate_route_distance(
                            merged_route["route_str"]
                        )
                    else:
                        # 如果不能合并，将当前路线作为新的起始路线
                        merged_route = current_route

                aggregated_routes[dc] = [merged_route]

        # 将聚合后的路线转换为列表
        final_routes = []
        for routes in aggregated_routes.values():
            final_routes.extend(routes)
        for route in final_routes:
            del route["route"]
        return final_routes

    def can_merge_routes(self, route1, route2):
        """
        检查两条路线是否可以合并，基于简单的距离和载重考虑。
        """
        # 检查无人机的载重限制和最大飞行距离
        max_capacity = MAX_CAPACITY  # 假设无人机的最大载重量为5
        total_capacity_needed = len(route1["orders"]) + len(route2["orders"])
        max_flight_distance = 20  # 无人机的最大飞行距离
        total_distance = route1["distance"] + route2["distance"]
        if (
            total_capacity_needed <= max_capacity
            and total_distance <= max_flight_distance
        ):
            return True
        return False

    def handle_non_urgent_orders(self):
        """
        处理非紧急和较紧急的订单，按照最大处理订单数限制。
        """
        non_urgent_orders = [
            order
            for order in self.current_orders
            if order["Priority"] in ("较紧急", "一般")
        ]
        routes = []
        selected_orders = (
            non_urgent_orders[:10] if len(non_urgent_orders) > 10 else non_urgent_orders
        )
        for order in selected_orders:
            route = self.assign_route_to_orders(order)
            routes.append(route)
        return routes

    def find_closest_distribution_center(self, delivery_point):
        """
        找到最近的配送中心。
        """
        # 实现逻辑来找到最近的配送中心
        min_distance = float("inf")
        closest_dc_index = None
        for dc_index in range(NUM_DC):
            dp_index = self.find_dp_index(delivery_point)
            distance = self.calculate_distance(dc_index, dp_index)
            if distance < min_distance:
                min_distance = distance
                closest_dc_index = dc_index
        return closest_dc_index

    def create_route_for_order(self, dc_index, order):
        """
        为单个订单创建配送路线。
        """
        dp_index = self.find_dp_index(order["Delivery Point"])
        route = {
            "distribution_center": dc_index + 1,
            "orders": [order["Order Id"]],
            "distance": self.calculate_distance(dc_index, dp_index),
            "route": [
                (dc_index + 1, "start"),
                (dp_index + 1, "end"),
            ],
        }
        return route

    def solve(self):
        simulation_duration = DURATION_TIME  # 持续时间为24小时，单位为分钟
        simulation_end_time = self.current_time + timedelta(minutes=simulation_duration)
        while self.current_time < simulation_end_time:
            self.get_orders()
            self.sort_orders_by_priority()
            self.plan_route()
            self.__increment_time()
            self.__update_orders()

    def __update_orders(self):
        """
        更新剩余订单的优先级，根据订单的截止时间与当前时间的差距。
        """
        current_time = self.current_time
        for order in self.current_orders:
            time_to_deadline = (order["Deadline"] - current_time).total_seconds() / 60
            if time_to_deadline <= 30:  # 假设30分钟内为高优先级
                order["Priority"] = 1
            elif time_to_deadline <= 60:  # 60分钟内为中等优先级
                order["Priority"] = 2


if __name__ == "__main__":
    order_file_path = "./orders_data.csv"
    dist_file_path = "./drone_delivery_data.csv"
    solver = Solver(order_file_path, dist_file_path)
    solver.solve()
