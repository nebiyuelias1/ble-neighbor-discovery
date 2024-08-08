import os
from stable_baselines3 import PPO

from lib.bluetooth_discovery_env import BluetoothDiscoveryEnv

model_dir = 'models/PPO'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
env = BluetoothDiscoveryEnv()
env.reset()

model  = PPO("MultiInputPolicy", env, verbose=1)

TIMESTEPS = 1000
iters = 0
while True:
    iters += 1
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{model_dir}/{TIMESTEPS*iters}")