import os
from stable_baselines3 import PPO

from lib.bluetooth_discovery_env import BluetoothDiscoveryEnv, ComputationMethod

model_dir = 'models/PPO'
logdir = "logs/PPO"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
env = BluetoothDiscoveryEnv(computation_method=ComputationMethod.SIMULATION)
env.reset()

model  = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000

for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", progress_bar=True)
    model.save(f"{model_dir}/{TIMESTEPS*i}")