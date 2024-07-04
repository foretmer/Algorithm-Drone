# order.py

import random
import itertools

order_id_generator = itertools.count()

def generate_orders():
    # Initialize an empty list to hold the new orders
    new_orders = []
    
    # Generate a random number of orders between 0 and 7
    for _ in range(random.randint(0, 9)):
        # Get the next unique order ID from the generator
        order_id = next(order_id_generator)
        
        # Randomly choose a priority for the order (0, 1, or 2)
        priority = random.choice([0, 1, 2])
        
        # Randomly choose a drop point for the order from the available drop points
        point = random.choice([2, 3, 4, 5, 6, 7, 8])
        
        # Append the order as a tuple (order_id, point, priority) to the list of new orders
        new_orders.append((order_id, point, priority))
    
    # Return the list of newly generated orders
    return new_orders

