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


