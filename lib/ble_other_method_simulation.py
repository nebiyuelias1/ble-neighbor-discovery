import random
import numpy as np

num_simulations = 10

class BLEOtherMethodSimulation:
    def __init__(self, scanning_interval, adv_interval, adv_window) -> None:
        # Parameters based on the description
        self.adv_interval_min = 20e-3  # 20 ms
        self.adv_interval_max = 10.24  # 10.24 s
        self.rd_max = 10e-3  # 10 ms
        self.T_min = 0  # scanWindow lower bound
        self.T_max = 10.24  # scanInterval upper bound
        self.T = scanning_interval  # Arbitrary choice of scan interval (modify as needed)
        self.Ts = 0.15  # Arbitrary choice of scan window (modify as needed)
        self.adv_interval = adv_interval
        self.Ta = adv_window

        # Advertisement parameters
        self.A_min = 80  # Min size of Adv_PDU in bits
        self.A_max = 376  # Max size of Adv_PDU in bits
        self.B = 1e-3  # Arbitrary choice for response wait time (modify as needed)

        # Simulation parameters
        self.simulation_time = 10000  # Total simulation time in seconds
        self.time_step = 125e-3  # Time step for the simulation in milliseconds

    # Function to simulate advertising events
    def advertise_event(self, Ta, rd):
        A = random.uniform(self.A_min, self.A_max)  # Random Adv_PDU size between min and max
        # adv_duration = A * 1e-6  # Convert to seconds (assuming 1 bit = 1 µs)
        adv_duration = 0.64
        rd_delay = random.uniform(0, rd)
        return Ta + rd_delay, self.Ta  # Return adv event time and duration

    # Function to simulate scanner's scanning behavior
    def scanner_event(self, current_time):
        if current_time % self.T < self.Ts:
            return True  # Scanner is active (within the scanWindow)
        return False  # Scanner is inactive

    def run(self):
        current_time = 0
        Ta = self.adv_interval
        discovery_time = None
        advertising_event_count = 1
        advertiser_next_event, adv_duration = self.advertise_event(Ta * advertising_event_count, self.rd_max)

        while current_time < self.simulation_time:
            # Scanner logic
            if self.scanner_event(current_time):
                # Check if the advertiser's event overlaps with the scanner's window
                if advertiser_next_event <= current_time <= advertiser_next_event + adv_duration:
                    discovery_time = current_time
                    break  # Discovery successful, stop the simulation

            # Update the advertiser's next event if the current one is done
            if current_time >= advertiser_next_event + adv_duration:
                advertising_event_count += 1
                advertiser_next_event, adv_duration = self.advertise_event(Ta * advertising_event_count, self.rd_max)

            current_time += self.time_step

        if discovery_time:
            print(f"Advertiser discovered at time {discovery_time:.3f} seconds")
            return discovery_time
        else:
            print("No discovery occurred within the simulation time.")
            return self.simulation_time

def run_simulation(scanning_interval, adv_interval, adv_window):
    latency_results = []

    arr = [0] * 100000
    for _ in range(num_simulations):
        
        simulation = BLEOtherMethodSimulation(scanning_interval,
                                              adv_interval=adv_interval,
                                              adv_window=adv_window)
        latency = simulation.run()

        latency_results.append(latency)

    beacon_distribution = []
    for i in range(1, len(arr)):
        if arr[i-1] > 0:
            beacon_distribution.extend([i] * arr[i-1])
    
    avg_latency = np.mean(latency_results)

    return {
        "avg_latency": avg_latency,
    }