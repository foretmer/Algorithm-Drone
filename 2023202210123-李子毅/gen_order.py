import csv
import random
from datetime import datetime, timedelta

from const import (
    DURATION_TIME,
    GRADE_ONE,
    GRADE_THREE,
    GRADE_TWO,
    NUM_ORDER_PER_POINT,
    START_TIME,
    UNIT_TIME,
)

# 定义订单优先级和时间参数
priority_levels = {
    "一般": GRADE_ONE,
    "较紧急": GRADE_TWO,
    "紧急": GRADE_THREE,
}  # 3小时转换为分钟


# 读取配送中心和卸货点的信息
def read_delivery_info(filename):
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        delivery_info = {}
        for row in reader:
            dc = f"DC{int(row[0])}"
            dp = f"DP{int(row[1])}"
            distance = float(row[2])
            if dc not in delivery_info:
                delivery_info[dc] = []
            delivery_info[dc].append((dp, distance))
    return delivery_info


# 生成订单数据
def generate_orders(
    delivery_info, num_orders_per_point, simulation_start_time, simulation_end_time
):
    orders = []
    order_id = 1
    current_time = simulation_start_time
    while current_time < simulation_end_time:
        for dc, dps in delivery_info.items():
            for _ in range(num_orders_per_point):
                priority = random.choice(list(priority_levels.keys()))
                deadline = current_time + timedelta(minutes=priority_levels[priority])
                dp, distance = random.choice(dps)
                orders.append(
                    {
                        "Order Id": order_id,
                        "Delivery Point": dp,
                        "Priority": priority,
                        "Deadline": deadline.strftime("%Y-%m-%d %H:%M:%S"),
                        "Created At": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                order_id += 1
        current_time += timedelta(minutes=UNIT_TIME)  # 每30分钟生成一次订单
    return orders


# 主逻辑
if __name__ == "__main__":
    # 读取配送中心和卸货点信息
    delivery_info = read_delivery_info("drone_delivery_data.csv")

    # 生成订单数据
    num_orders_per_point = NUM_ORDER_PER_POINT  # 假设每个配送中心每个时间步生成2个订单
    simulation_duration = DURATION_TIME  # 持续时间为24小时，单位为分钟

    simulation_start_time = datetime.strptime(START_TIME, "%Y-%m-%d %H:%M:%S")
    simulation_end_time = simulation_start_time + timedelta(minutes=simulation_duration)
    orders = generate_orders(
        delivery_info, num_orders_per_point, simulation_start_time, simulation_end_time
    )

    # 将订单数据存储到CSV文件中
    orders_filename = "orders_data.csv"
    with open(orders_filename, "w", newline="") as csvfile:
        fieldnames = [
            "Order Id",
            "Delivery Point",
            "Priority",
            "Deadline",
            "Created At",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for order in orders:
            writer.writerow(order)

    print(f"Orders data has been generated and saved to {orders_filename}")
