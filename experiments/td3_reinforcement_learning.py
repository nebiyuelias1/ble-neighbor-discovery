import os
from stable_baselines3 import TD3

from lib.bluetooth_discovery_env import BluetoothDiscoveryEnv, ComputationMethod

model_dir = 'models/TD3'
logdir = "logs/TD3"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
env = BluetoothDiscoveryEnv(computation_method=ComputationMethod.SIMULATION)
env.reset()

model  = TD3("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
iters = 0

for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TD3", progress_bar=True)
    model.save(f"{model_dir}/{TIMESTEPS*i}")