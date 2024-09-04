import random
import sys

# Example class to represent an order
class Order:
    def __init__(self, id, quantity, remaining_time):
        self.id = id
        self.quantity = quantity
        self.remaining_time = remaining_time

class Drone():
    def __init__(self, J, K, T, M, N, P, X, G):
        self.P = P
        self.X = X
        self.G = G
        self.W = {}
        self.Dm = {}
        self.Vm = []

        for p in P:
            p.V = {p}

        for x in X:
            temp = float('inf')
            nearest = None
            for p in P:
                dist = self.Distance(x, p, G)
                if dist < temp:
                    temp = dist
                    nearest = p
            nearest.V.add(x)

    # Define the Distance function using a distance matrix G
    def Distance(self, x, p, G):
        return G[x][p]
    
    # Generate orders for each unloading point
    def Generate_Orders(self):
        for x in self.X:
            # Simulating generating random orders
            num_orders = random.randint(1, 5)  # Generate 1 to 5 orders
            for i in range(num_orders):
                order_id = f"Order_{i+1}_{x}"
                quantity = random.randint(1, 10)  # Random quantity between 1 and 10
                remaining_time = random.randint(10, 30)  # Random remaining time between 10 and 30 minutes
                order = Order(order_id, quantity, remaining_time)
                # Store the order or handle it as needed (e.g., add to x.orders list)
                # For the sake of example, let's print the generated order
                print(f"Generated order: {order.id}, Quantity: {order.quantity}, Remaining Time: {order.remaining_time} minutes")

    # Update orders and their statuses
    def Update(self):
        for x in self.X:
            # Simulating updating orders for each unloading point
            for order in x.orders:
                order.remaining_time -= 1  # Decrease remaining time by 1 minute

                if order.remaining_time <= 0:
                    # Order delivery completed, process as needed
                    print(f"Order {order.id} delivered from {x}")

            # Filter out completed orders
            x.orders = [order for order in x.orders if order.remaining_time > 0]

    # Get N most urgent orders for a given drone
    def Get_N_Most_Urgent_Orders(self, drone):
        # Placeholder for selecting N most urgent orders
        # For example, sorting by remaining time and selecting the first N orders
        urgent_orders = sorted(drone.orders, key=lambda x: x.remaining_time)
        return urgent_orders[:4]  # Return the first N orders

    # Sort orders in descending order based on quantity
    def Sort(self, Dm):
        return sorted(Dm, key=lambda x: x.quantity, reverse=True)

    # Find the index of the most orders under remaining room
    def Find_Most_Orders_Under_Room(self, Vm, room):
        max_index = -1
        max_quantity = -sys.maxsize - 1
        for i, v in enumerate(Vm):
            if v.n <= room and v.n > max_quantity:
                max_index = i
                max_quantity = v.n
        return max_index

    # Function to find the minimum key value from the set of vertices not yet included in MST
    def min_key(self, key, mst_set, V):
        min_val = sys.maxsize
        min_index = -1
        for v in range(V):
            if key[v] < min_val and mst_set[v] == False:
                min_val = key[v]
                min_index = v
        return min_index

    # Function to construct MST and calculate TSP circuit length using MST heuristic
    def tsp_mst(self, adj_matrix):
        V = len(adj_matrix)  # Number of vertices

        # Initialize lists to store MST and TSP solution
        parent = [-1] * V
        key = [sys.maxsize] * V
        mst_set = [False] * V

        # Start with the first vertex as the root of MST
        key[0] = 0
        parent[0] = -1

        for _ in range(V - 1):
            u = self.min_key(key, mst_set, V)
            mst_set[u] = True

            for v in range(V):
                if 0 <= adj_matrix[u][v] < key[v] and not mst_set[v]:
                    parent[v] = u
                    key[v] = adj_matrix[u][v]

        # Construct TSP circuit from MST parent array
        tsp_circuit = []
        length = 0

        # Add edges from parent array to form TSP circuit
        for i in range(1, V):
            length += adj_matrix[i][parent[i]]
            tsp_circuit.append((parent[i], i))

        # Add the edge back to the starting vertex to complete the circuit
        length += adj_matrix[parent[V-1]][0]
        tsp_circuit.append((V-1, 0))

        # Prepare the route as a sequence of vertices
        route = []
        current = 0
        for edge in tsp_circuit:
            route.append(current)
            current = edge[1]
        route.append(0)  # Add the starting vertex to complete the route

        # Return TSP circuit length and route
        tsp_solution = {
            'length': length,
            'route': route
        }
        return tsp_solution

# Example usage
if __name__ == "__main__":
    # Example data
    J = 3  # Number of drone delivery centers
    K = 10  # Number of unloading points
    T = 20  # Time interval in minutes
    M = 100  # Number of orders
    N = 4  # Drone carrying capacity

    P = set()  # Set of delivery centers
    X = set()  # Set of unloading points
    G = {}     # Distance matrix

    # Call the function
    D = Drone(J, K, T, M, N, P, X, G)

    # Time loop
    while True:
        D.Generate_Orders()
        D.Update()

        for p in P:
            for v in p.V:
                if v == p:
                    continue
                if v.n >= N:
                    orders = D.Get_N_Most_Urgent_Orders(v)
                    D.W[len(D.W) + 1] = ([p, v, p], [N])
                    v.orders -= orders
                    v.n -= N

                for order in v.orders:
                    if 10 <= order.t <= 30:
                        D.Dm.add(order)

            D.Vm = D.Sort(D.Dm)
            while True:
                destinations = [p]
                quantity = []
                room = N

                while True:
                    idx = D.Find_Most_Orders_Under_Room(D.Vm, room)
                    if idx == -1:
                        break
                    v = D.Vm[idx]
                    tsp = D.tsp_mst(G)
                    if tsp['length'] <= 20:
                        destinations.append(v)
                        quantity.append(v.n)
                        D.Vm.remove(v)
                        room -= v.n

                D.W[len(D.W) + 1] = (destinations, quantity)
                if not D.Vm:
                    break
        # Print the result
        for d, (route, quantities) in W.items():
            print(f"Drone {d}: Route {route}, Quantities {quantities}")