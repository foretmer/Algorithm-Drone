# constants.py

DRONE_SPEED = 60  # Drone speed in km/h
TIME_INTERVAL = 30  # Time interval in minutes
MAX_CARGO = 3 # Max cargo capacity of a drone
MAX_DISTANCE = 20  # Max distance a drone can fly in one trip in km
DELIVERY_TIME = {0: 180, 1: 90, 2: 30}  # Delivery times in minutes for different priorities

NODES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
EDGES = [
    (0, 1, 5),
    (0, 4, 4),
    (0, 5, 7),
    (0, 3, 4),
    (0, 6, 7),
    (1, 2, 2),
    (1, 3, 6),
    (1, 6, 5),
    (2, 5, 4),
    (2, 6, 3),
    (3, 4, 3),
    (3, 6, 10),
    (4, 5, 2),
    (7, 1, 7),
    (7, 2, 4),
    (8, 3, 3),
    (8, 6, 5)
]
