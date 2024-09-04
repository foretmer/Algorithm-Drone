import random
import matplotlib.pyplot as plt
from utils import calculate_distance


def generate_points(num_delivery, num_drop, max_distance):
    dy_points = []
    dp_points = []

    # Generate delivery points
    for _ in range(num_delivery):
        x = random.randint(10, 30)
        y = random.randint(10, 30)
        dy_points.append((x, y))

    # Generate drop points ensuring each is within max_dist from at least one delivery point
    for _ in range(num_drop):
        while True:
            # Randomly select a delivery point
            dy_point = random.choice(dy_points)
            dp_point = (random.randint(0, 40), random.randint(0, 40))
            # Check if distance constraint is satisfied
            if any(calculate_distance(dp_point, dy_point) <= max_distance for dp in dy_points):
                dp_points.append(dp_point)
                break

    return dy_points, dp_points


def draw(points1, points2):
    x, y = [], []
    for xy in points1:
        x.append(xy[0])
        y.append(xy[1])
    plt.scatter(x, y, color='red', marker='*', s=100)  # 使用scatter函数绘制散点图

    x, y = [], []
    for xy in points2:
        x.append(xy[0])
        y.append(xy[1])
    plt.scatter(x, y, color='blue', s=50)  # 使用scatter函数绘制散点图

    # 设置图形属性
    plt.title('Address Scatter Plot with Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # 显示图形
    plt.grid(True)
    plt.savefig("AddressScatterPlot.png")
    plt.show()


# Example usage:
if __name__ == "__main__":
    num_delivery_points = 3
    num_drop_points = 8
    max_dist = 10
    delivery_points, drop_points = generate_points(num_delivery_points, num_drop_points, max_dist)
    draw(delivery_points, drop_points)
    print("Delivery Points:")
    for i, point in enumerate(delivery_points):
        print(f"DP{i + 1}: {point}")
    print("\nDrop Points:")
    for i, point in enumerate(drop_points):
        print(f"Drop {i + 1}: {point}")

