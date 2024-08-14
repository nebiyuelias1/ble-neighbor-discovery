import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum

from lib.math import analytical_latency_result, average_energy_consumption
from lib.ble_simulation import run_simulation

# Define low and high values for parameters
OMEGA_LOW = 0.1
OMEGA_HIGH = 1.9
L_LOW = 1.0
L_HIGH = 10.0
LAMBDA_LOW = 0.1
LAMBDA_HIGH = 1.0


class ComputationMethod(Enum):
    ANALYTICAL = 1
    SIMULATION = 2

class BluetoothDiscoveryEnv(gym.Env):
    def __init__(self, computation_method=ComputationMethod.ANALYTICAL) -> None:
        super(BluetoothDiscoveryEnv, self).__init__()

        self.computation_method = computation_method
        
        # Define action space: continuous range of omega, L, and lambda
        self.action_space = spaces.Box(low=np.array([OMEGA_LOW, L_LOW, LAMBDA_LOW]), high=np.array([OMEGA_HIGH, L_HIGH, LAMBDA_HIGH]), shape=(3,), dtype=np.float32)
        
        # Observations are dictionaries with the lost_device's and the scanner's parameters.
        self.observation_space = spaces.Dict(
            {
                # For lost device the parameters are omega(beacon width) and L(advertising interval)
                "lost_device": spaces.Box(low=np.array([OMEGA_LOW, L_LOW]), high=np.array([OMEGA_HIGH, L_HIGH]), shape=(2,), dtype=np.float32),
                # For scanner the parameter is lambda(scanning rate)
                "scanner": spaces.Box(low=LAMBDA_LOW, high=LAMBDA_HIGH, shape=(1,), dtype=np.float32),
            }
        )
        
        # Define initial state
        self.alpha = 0.5
        self.beta = 0.5
        
        # Define max steps per episode
        self.max_steps = 50  # Example value, adjust as needed
        self.current_step = 0  # Initialize current step
        
    def step(self, action):
        # Extract action values
        omega, L, lambda_ = action
        
        observation = self._get_observation(omega, L, lambda_)
        info = self._get_info(omega, L, lambda_)
        latency = info['latency']
        energy = info['energy']
        
        # Calculate reward (negative of objective):
        # In this Bluetooth neighbor discovery problem, 
        # the reward is expressed as the negative of the 
        # objective function, which is a common approach 
        # to transform a minimization problem into a maximization one.
        reward = -(self.alpha * latency + self.beta * energy)

        # Additional reward for significant improvements
        if latency < 5.0:
            reward += 10.0  # Bonus for very low latency
        if energy < 5.0:
            reward += 5.0  # Bonus for very low energy consumption

        # Episode termination condition
        done = latency < 5.0 and energy < 28.0 or self.current_step >= self.max_steps
        self.current_step += 1
        
        return observation, reward, done, False, info
    
    def _get_observation(self, omega, L, lambda_):
        return {
            "lost_device": np.array([omega, L]),
            "scanner": np.array([lambda_])
        }
        
    def _get_info(self, omega, L, lambda_):

        if self.computation_method == ComputationMethod.ANALYTICAL:
            # calculate latency
            latency = self.calculate_latency(omega=omega, L=L, lambda_=lambda_)
            # calculate energy usage
            energy = self.calculate_energy(omega=omega, L=L, lambda_=lambda_)
        elif self.computation_method == ComputationMethod.SIMULATION:
            result = run_simulation(period=L, beacon_duration=omega, rate=lambda_, include_energy_cost=True)
            latency = result['avg_latency']
            energy = result['avg_energy_cost']

        return {
            'latency': latency,
            'energy': energy
        }
    
    def calculate_latency(self, L, omega, lambda_):
        params = (L, omega, lambda_)
        latency = analytical_latency_result(params, 100, 100)
        return latency
    
    def calculate_energy(self, L, omega, lambda_):
        params = (L, omega, lambda_)
        energy_usage = average_energy_consumption(params)
        return energy_usage
    
    def reset(self, **kwargs):
        # Reset the environment to initial state
        super().reset(**kwargs)
        
        self.current_step = 0  # Initialize current step
        
        # Randomly initialize the state
        omega = np.random.uniform(OMEGA_LOW, OMEGA_HIGH)
        L = np.random.uniform(L_LOW, L_HIGH)
        lambda_ = np.random.uniform(LAMBDA_LOW, LAMBDA_HIGH)
        
        observation = self._get_observation(omega, L, lambda_)
        info = self._get_info(omega, L, lambda_)
        
        return observation, info
    
    def render(self, mode='human'):
        # Render the environment (if needed)
        pass