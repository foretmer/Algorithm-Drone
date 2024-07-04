from config import Item_Priority, Max_Order_Num, DP_position, Decision_Time_Interval
import random


class DP_Point:
    def __init__(self, location):
        self.location = location
        self.current_orders = []

    def generate_orders(self, start_time):
        num_orders = random.randint(0, Max_Order_Num)
        for _ in range(num_orders):
            priority = random.choice(Item_Priority)  # 随机生成订单优先级
            end_time_satisfy = start_time + priority  # 最晚送达时间
            self.current_orders.append(end_time_satisfy)
        self.current_orders.sort(key=lambda x: x)  # 按照优先级排序
        return

    def display_orders(self, start_time):
        print(f"在第{start_time}分钟时, 卸货点{self.location}"
              f"当前所有订单最迟送达时间: {self.current_orders}")


class DP_Points:
    def __init__(self, locations):
        self.dp_point_list = []
        for xy in locations:
            self.dp_point_list.append(DP_Point(xy))

    def generate_new_orders(self, start_time):
        for dp in self.dp_point_list:
            dp.generate_orders(start_time)

    def display_all_orders(self, start_time):
        for dp in self.dp_point_list:
            dp.display_orders(start_time)


# 示例用法
if __name__ == "__main__":
    dp_points = DP_Points(DP_position)
    now_time = 0
    while now_time <= 20:
        now_time = now_time + Decision_Time_Interval
        dp_points.generate_new_orders(now_time)
        dp_points.display_all_orders(now_time)
