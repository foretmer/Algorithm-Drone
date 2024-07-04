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
