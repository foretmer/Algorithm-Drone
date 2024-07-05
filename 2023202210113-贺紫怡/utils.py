import math
import random
import numpy as np

def get_dis(city1, city2):
    x1, y1 = city1[1], city1[2]
    x2, y2 = city2[1], city2[2]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return round(distance,3)

def generate_random_color():
    # 生成随机的 RGB 颜色值，范围从 0 到 1
    color = np.random.rand(3)
    # 将颜色转换为字符串形式
    return tuple(color)

