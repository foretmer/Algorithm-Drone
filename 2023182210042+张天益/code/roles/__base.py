from typing import Tuple

class Drone:
    def __init__(self, drone_id: int, capacity: int, speed: int):
        self.start_depot = None
        self.drone_id = drone_id
        self.capacity = capacity
        self.route = []
        self.load = 0
        self.speed = speed  # 1 unit distance per minute
        self.orders = []

class Dot:
    def __init__(self, x: float, y: float, id:int):
        self.x = x
        self.y = y
        self.id = id
        
class DropPoint(Dot):
    def __init__(self, x: float, y: float, id:int):
        super().__init__(x, y, id)
        self.depot = None
        
class DePot(Dot):
    def __init__(self, x: float, y: float, id:int):
        super().__init__(x, y, id)
        
class Order:
    def __init__(self, order_id: int,  destination: DropPoint, demand: int, time_window: Tuple[int, int], priority: int):
        self.order_id = order_id
        self.destination = destination
        self.demand = demand
        self.time_window = time_window
        self.priority = priority