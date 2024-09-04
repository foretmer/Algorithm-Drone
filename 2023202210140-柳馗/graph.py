# graph.py

import heapq
import matplotlib.pyplot as plt
import networkx as nx
from constants import EDGES, NODES, MAX_CARGO, MAX_DISTANCE

def initialize_graph(nodes, edges):
    # Initialize an empty adjacency list for each node in the graph
    graph = {node: [] for node in nodes}
    
    # Iterate over each edge defined by (u, v, w) where u and v are nodes and w is the weight
    for u, v, w in edges:
        # Append the edge (v, w) to the adjacency list of node u
        graph[u].append((v, w))
        # Append the edge (u, w) to the adjacency list of node v (because the graph is undirected)
        graph[v].append((u, w))
    
    # Return the constructed graph as an adjacency list
    return graph


def dijkstra(graph, start):
    # Initialize distances with infinity for all nodes, and set the start node distance to 0
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    
    # Initialize the priority queue with the start node and a distance of 0
    priority_queue = [(0, start)]

    while priority_queue:
        # Pop the node with the smallest distance from the priority queue
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # If the current distance is greater than the recorded distance, skip this node
        if current_distance > distances[current_node]:
            continue

        # Iterate over neighbors of the current node
        for neighbor, weight in graph[current_node]:
            # Calculate the distance to the neighbor
            distance = current_distance + weight
            
            # If the calculated distance is smaller than the recorded distance, update the distance
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                # Push the updated distance and neighbor to the priority queue
                heapq.heappush(priority_queue, (distance, neighbor))

    # Return the dictionary of distances from the start node to each other node
    return distances


def draw_graph():
    # Create a graph
    G = nx.Graph()
    
    # Define delivery centers and drop points
    delivery_centers = [0, 1]
    drop_points = [2, 3, 4, 5, 6, 7, 8]
    
    # Define labels for the nodes
    labels = {
        0: "Delivery Center 0",
        1: "Delivery Center 1",
        2: "Drop Point 2",
        3: "Drop Point 3",
        4: "Drop Point 4",
        5: "Drop Point 5",
        6: "Drop Point 6",
        7: "Drop Point 7",
        8: "Drop Point 8",
    }
    
    # Assign colors to nodes based on their type
    node_color = ["#36BA98" if node in delivery_centers else "#E9C46A" for node in NODES]

    # Add nodes and edges to the graph
    G.add_nodes_from(NODES)
    G.add_weighted_edges_from(EDGES)

    # Determine the positions for all nodes
    pos = nx.spring_layout(G)
    
    # Draw the graph with the specified node positions, labels, and colors
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_color, node_size=500, font_size=10, font_weight='bold')
    
    # Create a dictionary for edge labels showing the weights
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    
    # Draw the edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Add a legend for the node types
    plt.scatter([], [], c="#36BA98", label="Delivery Center", edgecolors="none", s=100)
    plt.scatter([], [], c="#E9C46A", label="Drop Point", edgecolors="none", s=100)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1)
    
    # Set the title of the plot
    plt.title("Drone Delivery Network", color="black")

    # Show the plot
    plt.show()


def choose_center(order, graph):
    """
    Selects the nearest delivery center based on the order destination.
    This function calculates the distances from two delivery centers to the order destination 
    and determines which center is closer using Dijkstra's algorithm.
    
    Args:
    order: Tuple containing order information, focusing on the destination.
    graph: Graph representation where distances are stored.

    Returns:
    Delivery center number (0 or 1) depending on which center is closer to the order destination.
    """
    # Calculate distances from delivery center 0 to all locations
    distances_from_0 = dijkstra(graph, 0)
    # Calculate distances from delivery center 1 to all locations
    distances_from_1 = dijkstra(graph, 1)

    # Compare distances from both centers to the order destination
    if distances_from_0[order[1]] <= distances_from_1[order[1]]:
        # If delivery center 0 is closer, return 0
        return 0
    else:
        # If delivery center 1 is closer, return 1
        return 1


def plan_path(center, orders, graph):
    """
    Plans the optimal paths from a center location to each order.
    
    Args:
    center: Starting location for path planning.
    orders: List of orders, each consisting of a location and demand.
    graph: Graph representation where distances are stored.

    Returns:
    A list of paths, each containing cargo (orders) and distance for each trip.
    """
    # List to store planned paths
    path = []
    # While there are still orders to process
    while orders:
        # Set current location to the center
        current_location = center
        # List to store orders for the current trip
        cargo = []
        # Accumulate distance for the current trip
        trip_distance = 0
        # Calculate shortest distances from current location
        distances = dijkstra(graph, current_location)
        
        # While cargo is not full and there are orders left
        while len(cargo) < MAX_CARGO and orders:
            # Choose the nearest order to the current location
            nearest_order = min(orders, key=lambda o: distances[o[1]])
            # Distance to the nearest order
            distance_to_order = distances[nearest_order[1]]
            # Check if within maximum trip distance
            if trip_distance + distance_to_order <= MAX_DISTANCE:
                # Add the nearest order to cargo
                cargo.append(nearest_order)
                # Remove the chosen order from the list
                orders.remove(nearest_order)
                # Update trip distance
                trip_distance += distance_to_order
                # Update current location
                current_location = nearest_order[1]
                # Recalculate shortest distances from current location
                distances = dijkstra(graph, current_location)
        
        # Add return distance to total trip distance
        trip_distance += distances[center]
        # Append cargo and distance for the current trip to the path list
        path.append((cargo, trip_distance))
    
    # Return the planned path list
    return path

