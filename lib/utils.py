import random
import numpy as np

# Initialize quorum grid
def initialize_quorum_grid(N):
    grid = np.zeros((N, N), dtype=int)
    grid = np.arange(0, N*N).reshape(N, N)
    return grid

def select_random_quorum(grid_size):
    active_column = random.randint(0, grid_size - 1)
    active_row = random.randint(0, grid_size - 1)
    return active_column, active_row

def is_overlapping(beacon_start_time, beacon_end_time, scanning_start_time, scanning_end_time):
    """
    Check if the beacon and scanning intervals overlap.

    Parameters:
    beacon_start_time (float): The start time of the beacon interval.
    beacon_end_time (float): The end time of the beacon interval.
    scanning_start_time (float): The start time of the scanning interval.
    scanning_end_time (float): The end time of the scanning interval.

    Returns:
    bool: True if the intervals overlap, False otherwise.
    """
    overlap_margin = (beacon_end_time - beacon_start_time) / 2
    return (beacon_start_time < scanning_end_time + overlap_margin) and (scanning_start_time < beacon_end_time + overlap_margin)
