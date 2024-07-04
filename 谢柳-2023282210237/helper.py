import numpy as np
from gurobipy import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

# Constants for drone parameters
WAIT_LIMIT = 10 / 60  # Defined the wait limit of drones to 10 minutes
DRONE_ENDURANCE = 30 / 60 # Defined the endurance of drones to 30 minutes
DRONE_TRAVEL_RADIUS = 20  # Defined the drone's travel radius to 20 miles


# Function to calculate the Euclidean distance between two cities.
def euclidean_distance(city1, city2):
    return np.linalg.norm(city1 - city2)


# Function to calculate a matrix of distances between the list of cities.
def calculate_distance_matrix(city_list):
    n = len(city_list)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = euclidean_distance(city_list[i], city_list[j])
    return dist_matrix

# Function to compute the total distance or time for a given tour using a distance or time matrix.
def calculate_total_distance_time(tour, matrix):
    total_distance_time = 0.0
    for i in range(len(tour) - 1):
        total_distance_time += matrix[tour[i], tour[i+1]]
    return total_distance_time

# Function to plot the given tour
import numpy as np
import plotly.graph_objects as go

def plot_tour(city_list, title, all_drone_tours=None):
    fig = go.Figure()

    # Convert city list to a numpy array for easier indexing
    city_coords = np.array(city_list)
    
    # Print the coordinates of each city
    print("City Coordinates:")
    for i, city in enumerate(city_list):
        print(f"City {i}: ({city[0]}, {city[1]})")

    # Plotting the cities
    fig.add_trace(go.Scatter(x=city_coords[:, 0], y=city_coords[:, 1], mode='markers', 
                             marker=dict(size=10, color='black'), name='City'))

    if all_drone_tours:
        # Print drone tour routes
        print("\nDrone Tour Routes:")
        for idx, drone_tour in enumerate(all_drone_tours):
            route = " -> ".join(map(str, drone_tour))
            print(f"Drone Tour {idx + 1}: {route}")

        # Plotting the first drone tour with legend
        first_drone_tour = all_drone_tours[0]
        fig.add_trace(go.Scatter(x=city_list[first_drone_tour, 0], y=city_list[first_drone_tour, 1], mode='lines', 
                                marker=dict(size=10, color='black'), line=dict(width=2, dash='dash'), 
                                name='Drone Tour', legendgroup='Drone Tours'))

        # Plotting the rest of the drone tours without legend but in the same legend group
        for drone_tour in all_drone_tours[1:]:
            fig.add_trace(go.Scatter(x=city_list[drone_tour, 0], y=city_list[drone_tour, 1], mode='lines', 
                                    marker=dict(size=10, color='black'), line=dict(width=2, dash='dash'), 
                                    showlegend=False, legendgroup='Drone Tours'))

    # Labeling the cities
    for i, city in enumerate(city_list):
        fig.add_annotation(go.layout.Annotation(
            x=city[0] + 0.35, y=city[1] + 0.35,
            xref="x", yref="y",
            text=str(i), showarrow=False, font=dict(size=12, color='orange')
        ))

    # Setting a title to the graph
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        xaxis=dict(constrain='domain', tickvals=list(range(0, 25, 2))),
        yaxis=dict(scaleanchor="x", tickvals=list(range(0, 25, 2))),
        title=title
    )

    # Displaying the plot
    fig.show()




# Function to get city indices that lie within a specified x and y range.
def get_cities_in_region(x_range, y_range, city_list):
    min_x, max_x = x_range
    min_y, max_y = y_range
    
    cities_in_region = [idx for idx, (x, y) in enumerate(city_list) if min_x <= x <= max_x and min_y <= y <= max_y]
    return cities_in_region

# Function to extract continuous sub-tours from a main tour that contain only the selected cities.
def get_subtours_in_region(tour, selected_cities):

    subtours = []
    current_subtour = []
    for idx in range(len(tour)-1):
        city = tour[idx]
        
        if city in selected_cities:
            current_subtour.append(city)
            
            if tour[idx+1] not in selected_cities:
                subtours.append(current_subtour)
                current_subtour = []

        elif current_subtour:
            subtours.append(current_subtour)
            current_subtour = []
    
    if current_subtour:
        subtours.append(current_subtour)

    return subtours

# Function to calculate matrices of times taken for both a truck and a drone to travel between cities.
def calculate_time_matrix(n, dist_matrix, DRONE_SPEED):
    # Initialize time matrices for drone and truck
    drone_time_matrix = np.zeros((n, n))

    # Calculate time matrices for drone and truck
    for i in range(n):
        for j in range(n):
            drone_time_matrix[i, j] = dist_matrix[i, j] / DRONE_SPEED
            
    return drone_time_matrix

# Function to estimate the maximum number of drones needed based on the number of cities.
def calculate_max_drones(city_count):
    return max(1, int(city_count * 1/6))

# Function to check if two segments of a tour overlap.
def segments_overlap(start1, end1, start2, end2, tour):
    idx_start1, idx_end1 = tour.index(start1), tour.index(end1)
    idx_start2, idx_end2 = tour.index(start2), tour.index(end2)
    return (idx_start1 <= idx_start2 <= idx_end1) or \
           (idx_start1 <= idx_end2 <= idx_end1) or \
           (idx_start2 <= idx_start1 <= idx_end2) or \
           (idx_start2 <= idx_end1 <= idx_end2)

# Function to find possible operations for drones to maximize the efficiency of deliveries.
def find_drone_operations(tour, drone_operations, dist_matrix, drone_time_matrix):
    min_drone_time = float('inf')
    best_start = None
    best_end = None
    best_drone_city = None

    for i in range(len(tour) - 2): # to ensure there are at least two more cities after the current one in the tour
        start_city = tour[i]
       

        for j in range(i + 2, len(tour)): # to ensure there's at least one city between the start and end for the drone to serve
            end_city = tour[j]
            
                    
            for k in range(i+1, j): # For a given start and end city, i and j, this loop iterates over all cities between them to find a potential drone city
                drone_city = tour[k]
                

                # Checking if the current drone segment was already used by another drone tour
                if any(segments_overlap(start_city, end_city, start, end, tour) for (start, end, _, _, _) in drone_operations):
                    continue
                
                
                # Drone's tour duration from start_city i to drone_city and then to end_city j
                drone_time = drone_time_matrix[start_city][drone_city] + drone_time_matrix[drone_city][end_city]
                # Compluting the distance travelled by the drone from start_city i to drone_city and then to end_city j
                drone_tour_length = dist_matrix[start_city][drone_city] + dist_matrix[drone_city][end_city]


                 # Check drone constraints: 
                #   1. drone's tour distance should not be greater than its travel radius
                #   2. drone's travel duration should not be greater than its endurance
                #   3. drone's waiting time for the truck should be less than its wait limit 
                #   4. drone's waiting time should be positive, if its negative that means truck arrives before the drone
                if drone_tour_length > DRONE_TRAVEL_RADIUS or drone_time >= DRONE_ENDURANCE:
                    continue
                
                # Calculating the Savings, if positve that means using drone for that segment is feasible


                # higher the savings, more time will be saved by using the drone for that segment
                if drone_time <= min_drone_time:
                    best_start = start_city
                    best_end = end_city
                    best_drone_city = drone_city

    return best_start, best_end, best_drone_city, min_drone_time

# Function to adjust a tour generated using the two-opt heuristic based on drone operations.
def adjust_two_opt_tour_for_drone_operations(tour, drone_operations):
    # Extract all drone cities from the operations
    drone_cities = {operation[2] for operation in drone_operations}

    adjusted_tour = [tour[0]]  # Start with the first city of the tour
    for i in range(1, len(tour)):
        if tour[i] not in drone_cities:
            adjusted_tour.append(tour[i])

    return adjusted_tour

# Function to reconstruct a tour based on the solution matrix obtained from an optimization model.
def get_tour_from_solution_matrix(y, start_city):
    tour = []
    current_city = start_city
    next_city = None

    # Continuously try to find the next city in the tour until reaching the start city or no city is found.
    while True:
        # Append the current city to the tour.
        tour.append(current_city)
        
        # Fetching the next possible cities from the solution matrix y, based on:
        # 1. The city is connected to the current city.
        # 2. The city has not been visited yet (not in the tour).
        possible_next_cities = [j for (i, j) in y.keys() if i == current_city and j not in tour]
        
        # Iterate through each possible cities.
        for j in possible_next_cities:
            # If there is an edge present between the current city and city j in our solution matrix, select city j as the next city in the tour.
            if y[current_city, j].x > 0.5:
                next_city = j
                break

        # If we reached the starting city again or no next city found, end the tour
        if next_city is None or next_city == start_city:
            break

        # Move to the next city and reset the next_city variable for the upcoming iteration.
        current_city = next_city
        next_city = None

    return tour

# Function to Plot bar plots for comparison of distances and durations by methods.
def plot_comparison(table_data):
    
    # Splitting the table_data for better visualization
    methods = [row[0] for row in table_data]
    distances = [float(row[1].split()[0]) for row in table_data]
    durations = [float(row[2].split()[0]) for row in table_data]

    # Setting up seaborn
    sns.set_style("whitegrid")

    # Create a bar plot for distances
    plt.figure(figsize=(6, 6), dpi=100)
    ax1 = sns.barplot(y=methods, x=distances, palette="viridis", orient="h")
    plt.title("Comparison of Distances by Methods")
    plt.xlabel("Distance (miles)")
    
    # Annotate each bar with its value
    for p in ax1.patches:
        width = p.get_width()
        plt.text(width + 0.2, p.get_y() + p.get_height()/2. + 0.2, '{:1.2f}'.format(width), ha="center")
    
    plt.tight_layout()
    plt.show()

    # Create a bar plot for durations
    plt.figure(figsize=(6, 6), dpi=100)
    ax2 = sns.barplot(y=methods, x=durations, palette="rocket", orient="h")
    plt.title("Comparison of Durations by Methods")
    plt.xlabel("Duration (minutes)")
    
    # Annotate each bar with its value
    for p in ax2.patches:
        width = p.get_width()
        plt.text(width + 0.2, p.get_y() + p.get_height()/2. + 0.2, '{:1.2f}'.format(width), ha="center")
    
    plt.tight_layout()
    plt.show()