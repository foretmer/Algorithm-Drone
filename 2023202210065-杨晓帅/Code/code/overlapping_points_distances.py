"""

@Author: So What
@Time: 2024/6/30 18:52
@File: overlapping_points_distances.py

"""
import numpy as np

import numpy as np


def calculate_distances(array1, array2):
    """
    Calculate the distances between each pair of points from two 2D NumPy arrays.

    Parameters:
    array1 (np.ndarray): First input 2D NumPy array with shape (m, 2).
    array2 (np.ndarray): Second input 2D NumPy array with shape (n, 2).

    Returns:
    np.ndarray: A 2D NumPy array with shape (m, n) containing the distances between each pair of points.
    """
    # Get the number of points in each array
    m = array1.shape[0]
    n = array2.shape[0]

    # Initialize an empty array to store the distances
    distances = np.zeros((m, n))

    # Calculate the distances
    for i in range(m):
        for j in range(n):
            distances[i, j] = np.linalg.norm(array1[i] - array2[j])

    return distances


# Example usage
if __name__ == "__main__":
    # Example input arrays
    array1 = np.array([[31.38651741,15.63868617],
                       [34.77839174,15.85826984],
                       [38.44347605,21.76962673],
                       [41.89185044,18.56368174]],)

    array2 = np.array([[40.78822419,12.58522706],
                       [33.72449907,23.95258972],])

    # Calculate distances
    distance_matrix = calculate_distances(array1, array2)

    # Print the result
    print("Distance Matrix:")
    print(distance_matrix)


