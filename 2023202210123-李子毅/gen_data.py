import csv
import random
from const import NUM_DC, NUM_DP, MAX_DISTANCE


# 配送中心和卸货点的编号范围
distribution_center_ids = range(1, NUM_DC + 1)
delivery_point_ids = range(NUM_DC + 1, NUM_DC + NUM_DP + 1)


# 随机生成配送中心和卸货点之间的距离
def generate_distances(dc_ids, dp_ids, max_distance):
    distances = []
    # 生成配送中心和卸货点之间的距离
    for dc in dc_ids:
        for dp in dp_ids:
            distance = random.uniform(1, max_distance)
            distances.append([dc, dp, distance])
    # 生成卸货点之间的距离
    for dp1 in dp_ids:
        for dp2 in dp_ids:
            if dp1 < dp2:  # 避免重复计算
                distance = random.uniform(1, max_distance)
                distances.append([dp1, dp2, distance])
    return distances


# 生成所有点之间的距离数据
all_distances = generate_distances(
    distribution_center_ids, delivery_point_ids, MAX_DISTANCE
)

# 将数据保存为CSV
with open("drone_delivery_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["From", "To", "Distance"])
    for row in all_distances:
        writer.writerow(row)

print("Data has been generated and saved to drone_delivery_data.csv")
