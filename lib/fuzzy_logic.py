import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def get_recommended_quorum_size(battery_level_input, 
                                traffic_load_input,
                                device_role_input):
    """
    Calculate the recommended quorum size based on the given battery level and traffic load inputs.
    Parameters:
    battery_level_input (float): The current battery level input for the fuzzy logic system.
    traffic_load_input (float): The current traffic load input for the fuzzy logic system.
    device_role_input (int): The role of the device, either advertiser=0/scanner=1
    Returns:
    int: The recommended quorum size calculated by the fuzzy logic system.
    """     
    # Define fuzzy variables
    battery_level = ctrl.Antecedent(np.arange(0, 101, 1), 'battery_level')
    traffic_load = ctrl.Antecedent(np.arange(0, 11, 1), 'traffic_load')
    device_role = ctrl.Antecedent(np.arange(0, 2, 1), 'device_role')

    quorum_size = ctrl.Consequent(np.arange(0, 11, 1), 'quorum_size')

    # Membership functions
    battery_level['very_low'] = fuzz.trapmf(battery_level.universe, [0, 0, 10, 30])
    battery_level['low'] = fuzz.trimf(battery_level.universe, [10, 30, 50])
    battery_level['medium'] = fuzz.trimf(battery_level.universe, [30, 50, 70])
    battery_level['high'] = fuzz.trapmf(battery_level.universe, [50, 70, 100, 100])

    traffic_load['low'] = fuzz.trapmf(traffic_load.universe, [0, 0, 2, 4])
    traffic_load['medium'] = fuzz.trimf(traffic_load.universe, [2, 5, 8])
    traffic_load['high'] = fuzz.trapmf(traffic_load.universe, [6, 8, 10, 10])

    # Membership functions for device_role
    device_role['advertiser'] = fuzz.trimf(device_role.universe, [0, 0, 1])
    device_role['scanner'] = fuzz.trimf(device_role.universe, [1, 2, 2])

    quorum_size['small'] = fuzz.trapmf(quorum_size.universe, [0, 0, 1, 4])
    quorum_size['moderate'] = fuzz.trimf(quorum_size.universe, [2, 5, 8])
    quorum_size['large'] = fuzz.trapmf(quorum_size.universe, [6, 8, 10, 10])

    # Define rules
    rule1 = ctrl.Rule(battery_level['very_low'] & device_role['advertiser'], quorum_size['large'])
    rule2 = ctrl.Rule(battery_level['low'] & device_role['advertiser'], quorum_size['moderate'])
    rule3 = ctrl.Rule(battery_level['medium'] & device_role['advertiser'], quorum_size['moderate'])
    rule4 = ctrl.Rule(battery_level['high'] & device_role['advertiser'], quorum_size['small'])
    

    # Create control system
    quorum_ctrl = ctrl.ControlSystem([rule1, 
                                      rule2,
                                      rule3,
                                      rule4,
                                    ])
    quorum_sim = ctrl.ControlSystemSimulation(quorum_ctrl)

    # Input values
    quorum_sim.input['battery_level'] = battery_level_input
    # quorum_sim.input['traffic_load'] = traffic_load_input
    quorum_sim.input['device_role'] = device_role_input

    # Compute output
    quorum_sim.compute()
    recommended_quorum_size = round(quorum_sim.output['quorum_size'])
    
    return recommended_quorum_size

# Example usage
# battery_level_input = 30
# traffic_load_input = 7
# recommended_quorum_size = get_recommended_quorum_size(battery_level_input, traffic_load_input)
# print(f"Recommended Quorum Size: {recommended_quorum_size}")
