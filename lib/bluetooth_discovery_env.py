import gymnasium as gym
from gymnasium import spaces
import numpy as np

from lib.math import analytical_latency_result, average_energy_consumption

# Define low and high values for parameters
OMEGA_LOW = 0.1
OMEGA_HIGH = 2.5
L_LOW = 1.0
L_HIGH = 10.0
LAMBDA_LOW = 0.1
LAMBDA_HIGH = 1.0

class BluetoothDiscoveryEnv(gym.Env):
    def __init__(self) -> None:
        super(BluetoothDiscoveryEnv, self).__init__()
        
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
        reward = - (self.alpha * latency + self.beta * energy)

        # Episode termination condition
        done = latency < 20.0  # Example condition
        
        return observation, reward, done, False, info
    
    def _get_observation(self, omega, L, lambda_):
        return {
            "lost_device": np.array([omega, L]),
            "scanner": np.array([lambda_])
        }
        
    def _get_info(self, omega, L, lambda_):
        # calculate latency
        latency = self.calculate_latency(omega=omega, L=L, lambda_=lambda_)
        
        # calculate energy usage
        energy = self.calculate_energy(omega=omega, L=L, lambda_=lambda_)

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