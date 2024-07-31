import gymnasium as gym
from gymnasium import spaces
import numpy as np

from lib.math import analytical_latency_result, average_energy_consumption

class BluetoothDiscoveryEnv(gym.Env):
    def __init__(self) -> None:
        super(BluetoothDiscoveryEnv, self).__init__()
        
        # Define action space: continuous range of scanning rates
        self.action_space = spaces.Box(low=0.1, high=1.0, shape=(1,), dtype=np.float32)
        
        # Define state space: scanning rate, latency, energy usage
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, float('inf')]), dtype=np.float32)
        
        # Initial state
        self.state = np.array([1.0, 0.0])  # [lambda, initial latency]
        
        # Parameters
        self.L = 1.0  # Advertising interval
        self.omega = 0.1  # Beacon width
        self.alpha = 1.0  # Weight for latency
        self.beta = 0.5  # Weight for energy
        self.c1 = 1.0  # Energy cost constant 1
        self.c2 = 0.5  # Energy cost constant 2
        
    def step(self, action):
        lambda_ = action[0]
        
        # calculate latency
        latency = self.calculate_latency(lambda_)
        
        # calculate energy usage
        energy = self.calculate_energy(lambda_)
        
        # Calculate reward (negative of objective):
        # In this Bluetooth neighbor discovery problem, 
        # the reward is expressed as the negative of the 
        # objective function, which is a common approach 
        # to transform a minimization problem into a maximization one.
        reward = - (self.alpha * latency + self.beta * energy)
        
        # Update state
        self.state = np.array([lambda_, latency])

        # Episode termination condition
        done = latency < 20.0  # Example condition
        
        return self.state, reward, done, False, {}
    
    def calculate_latency(self, lambda_):
        params = (self.L, self.omega, lambda_)
        latency = analytical_latency_result(params, 1000, 1000)
        return latency
    
    def calculate_energy(self, lambda_):
        params = (self.L, self.omega, lambda_)
        energy_usage = average_energy_consumption(params)
        return energy_usage
    
    def reset(self, **kwargs):
        # Reset the environment to initial state
        self.state = np.array([1.0, 0.0])
        return self.state, {}
    
    def render(self, mode='human'):
        # Render the environment (if needed)
        pass