import numpy as np

# Define constants
num_simulations = 1000

class LostDevice:
    def __init__(self, env, period, beacon_duration, beacon_events):
        self.env = env
        self.period = period
        self.beacon_duration = beacon_duration
        self.beacon_events = beacon_events
        self.action = env.process(self.send_beacon())

    def send_beacon(self):
        while True:
            
            yield self.env.timeout(self.period)
            now = self.env.now
            self.beacon_events.append(now)
            
            
class ScannerDevice:
    def __init__(self, env, beacon_duration, rate, latency_results, beacon_events):
        self.env = env
        self.beacon_duration = beacon_duration
        self.rate = rate
        self.latency_results = latency_results
        self.beacon_events = beacon_events
        self.action = env.process(self.scan_beacon())

    def scan_beacon(self):
        range_entrance_start_time = self.env.now

        while True:
            yield self.env.timeout(np.random.exponential(scale=1/self.rate))
            current_time = self.env.now

            if len(self.beacon_events) > 0:
                beacon_time = self.beacon_events.pop()
                beacon_end_time = beacon_time + self.beacon_duration
                if beacon_time <= current_time <= beacon_end_time:
                    latency = current_time - range_entrance_start_time
                    self.latency_results.append(latency)
                    break
                
class Simulation:
    """
    A class to represent a simulation of neighbor discovery of BLE devices.
    """
    def __init__(self, rate, beacon_duration, beacon_period) -> None:
        """This constructor initializes the simulation parameters.

        Args:
            rate (float): the rate of scanning events
            beacon_duration (float): The duration of the beacon event
            beacon_period (float): The time period between two beacon events
        """
        self.rate = rate
        self.beacon_duration = beacon_duration
        self.beacon_period = beacon_period
        
    def run(self):
        # The time of the kth scanning event
        y_k = 0
        # The index of the beacon event
        n_i = 1
        
        while True:
            # The time since the last scanning event
            # Start by drawing a random number from the exponential distribution
            # Then keep adding this to y_k until it exceeds the beacon event bounds.
            x_i = np.random.exponential(scale=1/self.rate)
            y_k += x_i
            
            # The left and right bounds of the beacon event
            a = n_i * self.beacon_period - self.beacon_duration / 2
            b = n_i * self.beacon_period + self.beacon_duration / 2
            
            # If the scanning event happens before the beacon event
            # then we need to keep generating beacon events.
            while y_k > b:
                n_i += 1
                a = n_i * self.beacon_period - self.beacon_duration / 2
                b = n_i * self.beacon_period + self.beacon_duration / 2

            # Discovery has happened
            if a <= y_k <= b:
                return y_k
                
def run_simulation(period, beacon_duration, rate):
    latency_results = []

    for _ in range(num_simulations):
        simulation = Simulation(rate, beacon_duration, period)
        latency_results.append(simulation.run())

    avg_latency = np.mean(latency_results)
    return avg_latency