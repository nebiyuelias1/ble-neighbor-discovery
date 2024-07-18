import numpy as np
import matplotlib.pyplot as plt

# Define constants
num_simulations = 100000
energy_cost_of_beacon = 0.01
energy_cost_of_scanning = 2.0

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
        
    def run(self, arr, include_energy_cost=False):
        # The time of the kth scanning event
        y_k = 0
        # The index of the beacon event
        n_i = 1
        arr_y_ks = []
        total_energy_cost = 0
        
        while True:
            # The time since the last scanning event
            # Start by drawing a random number from the exponential distribution
            # Then keep adding this to y_k until it exceeds the beacon event bounds.
            x_i = np.random.exponential(scale=1/self.rate)
            # print(f'X_i: {x_i}')
            y_k += x_i
            arr_y_ks.append(y_k)
            
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
                arr[n_i-1] += 1
                # print(f'Discovery has happened after beacon: {n_i}, time: {y_k}, a: {a}, b: {b}')
                # draw_neighbor_discovery_process(self.beacon_period, self.beacon_duration, n_i, arr_y_ks)
                total_energy_cost = n_i * energy_cost_of_beacon * \
                    self.beacon_duration + \
                    (len(arr_y_ks) * energy_cost_of_scanning)
                if (include_energy_cost):
                    return y_k, total_energy_cost
                
                return y_k


def draw_neighbor_discovery_process(L, omega, n_beacons_top, t_scans):
    t_top = [i*L for i in range(n_beacons_top+1)]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw top timeline beacons
    ax.vlines(t_top, ymin=0.5, ymax=1.0, color='blue', label='Advertising Beacons')
    ax.plot(t_top, np.ones_like(t_top) * 0.75, 'bo')

    # Draw bottom timeline beacons
    ax.vlines(t_scans, ymin=0.0, ymax=0.5, color='red', label='Scanning Events')
    ax.plot(t_scans, np.ones_like(t_scans) * 0.25, 'ro')

    # Highlight nth beacon
    ax.axvspan(t_top[-1] - omega / 2, t_top[-1] + omega / 2, color='green', alpha=0.3)

    # Add labels and grid
    ax.set_xlabel('Time (t)')
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(['Scanning Events', 'Advertising Beacons'])
    # ax.set_xticks(np.arange(0, max(t_top[-1], t_scans[-1]) + omega, omega))
    ax.grid(True)

    plt.legend()
    plt.title(f'Bluetooth Neighbor Discovery, L={L}, ω={omega}, n={n_beacons_top}')
    plt.show()


def calculate_ci_bootstrapping(data, confidence_level=0.95):
    """
    Calculates confidence interval for average latency using bootstrapping.

    Args:
        data: List of simulated latency values for a specific L value.
        num_iterations: Number of resampled datasets to generate (default 1000).
        confidence_level: Desired confidence level for the interval (default 0.95).

    Returns:
        Tuple containing lower and upper confidence limit for the average latency.
    """
    resampled_latencies = []
    for _ in range(num_simulations):
        resample = np.random.choice(data, size=len(data), replace=True)  # Resample with replacement
        resampled_latencies.append(np.mean(resample))  # Calculate average latency for resampled set

    percentiles = np.percentile(resampled_latencies, [confidence_level * 100 / 2, 100 - confidence_level * 100 / 2])
    return percentiles[0], percentiles[1]  # Lower and upper confidence limit


def calculate_ci(latency_results):
    """
    Calculate the confidence interval for the average latency using the formula: ci = avg_latency ± z * std_dev / sqrt(n)
    
    Params:
        latency_results: List of latency values from the simulation
        
    Returns:
        Tuple containing the lower and upper confidence interval
    """
    # Calculate confidence interval using formula and std deviation
    std_dev = np.std(latency_results)
    z_value = 1.96  # 95% confidence interval
    avg_latency = np.mean(latency_results)
    lower_ci = avg_latency - z_value * (std_dev / np.sqrt(num_simulations))
    upper_ci = avg_latency + z_value * (std_dev / np.sqrt(num_simulations))
    
    return lower_ci, upper_ci

def draw_histogram(arr, period, beacon_duration, rate):
    """Draw histogram plot of the latency results.

    Args:
        latency_results (array): array of latency values
        period (number): The period of the beacon event
        beacon_duration (number): The duration of the beacon event
        rate (number): The rate of scanning events
    """
    # Plot the histogram of the latency results
    plt.hist(arr, bins=100)
    plt.xlabel("Latency")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Latency Results with L={period}, ω={beacon_duration}, λ={rate}")
    plt.show()
                
def run_simulation(period, beacon_duration, rate, include_energy_cost=False):
    latency_results = []
    energy_cost_results = []

    arr = [0] * 100000
    for _ in range(num_simulations):
        
        simulation = Simulation(rate, beacon_duration, period)
        if include_energy_cost:
            latency, energy_cost = simulation.run(arr, include_energy_cost)
            energy_cost_results.append(energy_cost)
        else:
            latency = simulation.run(arr)

        latency_results.append(latency)

    beacon_distribution = []
    for i in range(1, len(arr)):
        if arr[i-1] > 0:
            beacon_distribution.extend([i] * arr[i-1])
    
    avg_latency = np.mean(latency_results)
    avg_energy_cost = np.mean(energy_cost_results) if include_energy_cost else None
    
    lower_ci, upper_ci = calculate_ci(latency_results)

    return {
        "avg_latency": avg_latency,
        "avg_energy_cost": avg_energy_cost,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci
    }