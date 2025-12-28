#!/usr/bin/env python
# coding: utf-8

# # 1. Getting Up and Running

# In[6]:


get_ipython().system('pip install vizdoom')


# In[7]:


import os 
os.getcwd()


# In[2]:


cd Vizdoom


# In[ ]:


get_ipython().system('git clone https://github.com/mwydmuch/ViZDoom')


# In[8]:


# Import vizdoom for game env
from vizdoom import * 
# Import random for action sampling
import random
# Import time for sleeping
import time 
# Import numpy for identity matrix
import numpy as np


# In[ ]:


# Setup Game
game = DoomGame()
game.load_config('ViZDoom/scenarios/deadly_corridor_s1.cfg')
game.init()


# In[112]:


game.close()


# In[5]:


# This is the set of actions we can take in the environment
actions = np.identity(7, dtype=np.uint8)


# In[ ]:


game.make_action(actions[2])


# In[ ]:


state = game.get_state()


# In[ ]:


state.game_variables


# In[5]:


health,damage_taken,hitcount,ammo = state.game_variables


# In[ ]:


from vizdoom import GameVariable
# loop through episodes
episodes = 5
for episode in range(episodes): 
    # Create a new episode or game 
    game.new_episode()
    # Check the game isn't done 
    while not game.is_episode_finished(): 
        # Get the game state 
        state = game.get_state()
        # Get the game image 
        img = state.screen_buffer
        # Get the game variables - ammo
        info = state.game_variables
        # Take an action
        reward = game.make_action(random.choice(actions), 4)
        killCount = game.get_game_variable(GameVariable.KILLCOUNT)
        print(killCount)
        # Print rewward 
        #print('reward:', reward) 
        time.sleep(0.02)
    print('Result:', game.get_total_reward())
    time.sleep(2)


# # 2- Converting to Gym Env

# In[3]:


# Import environment base class from OpenAI Gym
from gym import Env
# Import gym spaces 
from gym.spaces import Discrete, Box
# Import opencv 
import cv2


# In[4]:


class VizDoomGym(Env):
    def __init__(self, render=False, config='ViZDoom/scenarios/deadly_corridor_s1.cfg'):
        super().__init__()
        # Game and state setup
        self.game = DoomGame()
        self.game.load_config(config)
        if not render:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        self.game.init()
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(7)
        self.damage_taken = 0
        self.killcount=0
        self.hitcount = 0
        self.ammo = 52

    def step(self, action):
        actions = np.identity(7)
        movement_reward = self.game.make_action(actions[action], 4)
        reward = 0
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            game_variables = self.game.get_state().game_variables
            health, killcount, damage_taken, hitcount, ammo = game_variables

            # Experiment with these values!
            # Calculate reward deltas
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            killcount_delta = killcount - self.killcount
            self.killcount = killcount
            
            reward = movement_reward + damage_taken_delta*15 + hitcount_delta*800 + killcount_delta*2000  + ammo_delta*-0.5

            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0
        info = {"info": info}
        done = self.game.is_episode_finished()
        return state, reward, done, info

    # Other methods omitted for brevity...
  # Define how to render the game or environment 
    def render(): 
        pass
    
    # What happens when we start a new game 
    def reset(self): 
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)
    
    # Grayscale the game frame and resize it 
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state
    
    # Call to close down the game
    def close(self): 
        self.game.close()


# In[10]:


env = VizDoomGym(render=True)


# In[ ]:


env.step(6)


# In[ ]:


env.step(2)


# In[ ]:


state = env.reset()


# # 3. View Game State

# In[ ]:


env.reset()


# In[11]:


env.close()


# In[12]:


# Import Environment checker
from stable_baselines3.common import env_checker


# In[ ]:


env_checker.check_env(env)


# # 4. View State

# In[ ]:


get_ipython().system('pip install matplotlib')


# In[13]:


from matplotlib import pyplot as plt


# In[ ]:


plt.imshow(cv2.cvtColor(state, cv2.COLOR_BGR2RGB))


# # 4. Setup Callback

# In[ ]:


get_ipython().system(' pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[ ]:


get_ipython().system('pip install stable-baselines3[extra]')


# In[14]:


# Import os for file nav
import os 
# Import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback


# In[25]:


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


# In[26]:


CHECKPOINT_DIR = './train/train_corridor3'
LOG_DIR = './logs/log_corridor3'


# In[27]:


callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)


# # 5. Train our Model Using Curriculum

# In[28]:


# import ppo for training
from stable_baselines3 import PPO


# In[29]:


model.load('./train/train_corridor1/best_model_600000.zip')


# In[30]:


model = PPO.load('./train/train_corridor1/best_model_400000')


# In[71]:


# Non rendered environment
env = VizDoomGym(config='ViZDoom/scenarios/deadly_corridor_s1.cfg')


# In[102]:


#model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)


# In[563]:


model.learn(total_timesteps=300000, callback=callback)


# In[103]:


env = VizDoomGym(config='ViZDoom/scenarios/deadly_corridor_s1.cfg')
model.set_env(env)
model.learn(total_timesteps=100000, callback=callback)


# # 6. Test the Model

# In[31]:


from stable_baselines3.common.evaluation import evaluate_policy


# In[32]:


env = VizDoomGym(render=True)


# In[34]:


env.close()


# In[33]:


mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)


# In[ ]:


mean_reward


# In[ ]:


for episode in range(5):
    obs = env.reset()
    done=False
    total_reward = 0
    while not done:
        action, _ =model.predict(obs)
        obs,reward,done,info=env.step(action)
        time.sleep(0.20)
        total_reward+=reward
    print('Total reward for episode {} is {}'.format(total_reward,episode))
    time.sleep(2)
        


# In[125]:



# Assuming you have collected data during training
timestamps = [0,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,200000,300000,400000]  # List of timestamps
enemy_kills = [0,0,0,1,1,2,2,1,2,1,2,3,4,4]  # List of corresponding enemy counts

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(timestamps, enemy_kills, color='red', marker='o', linestyle='-')
plt.title('Kills of Enemies vs Timestamp')
plt.xlabel('Timestamp')
plt.ylabel('Kills of Enemies')
plt.grid(True)
plt.show()


# In[2]:


import matplotlib.pyplot as plt

# Provided data
data = [[1703931550.105779, 8192, 207.2820587158203], [1703931582.6805153, 16384, 182.95506286621094], [1703931615.441619, 24576, 156.4600067138672], [1703931648.2357233, 32768, 149.8300018310547], [1703931681.1745083, 40960, 99.18000030517578], [1703931713.9667296, 49152, 98.88999938964844], [1703931746.8779507, 57344, 106.51000213623047], [1703931779.5474904, 65536, 93.16000366210938], [1703931812.2826276, 73728, 115.58999633789062], [1703931844.8429425, 81920, 137.88999938964844], [1703931877.370046, 90112, 169.86000061035156], [1703931909.8691456, 98304, 185.5], [1703931942.3847542, 106496, 149.99000549316406], [1703931974.5528765, 114688, 122.72000122070312], [1703932007.0530589, 122880, 117.72000122070312], [1703932039.5742335, 131072, 102.51000213623047], [1703932072.0054526, 139264, 105.25], [1703932104.6035993, 147456, 116.97000122070312], [1703932137.196046, 155648, 113.2300033569336], [1703932169.714128, 163840, 92.45999908447266], [1703932202.207281, 172032, 94.93000030517578], [1703932234.6917255, 180224, 86.62999725341797], [1703932267.176712, 188416, 80.22000122070312], [1703932299.8298483, 196608, 84.88999938964844], [1703932332.4164426, 204800, 76.26000213623047], [1703932364.9803915, 212992, 73.87999725341797], [1703932397.5695689, 221184, 74.62999725341797], [1703932430.0804992, 229376, 72.04000091552734], [1703932462.5116606, 237568, 73.20999908447266], [1703932494.9359422, 245760, 75.87000274658203], [1703932527.4242802, 253952, 77.44000244140625], [1703932559.8503745, 262144, 80.04000091552734], [1703932592.5054398, 270336, 84.62000274658203], [1703932625.0245128, 278528, 86.47000122070312], [1703932657.646088, 286720, 86.45999908447266], [1703932690.225402, 294912, 86.45999908447266], [1703932722.9261813, 303104, 84.16999816894531], [1703932755.564426, 311296, 86.94000244140625], [1703932788.0665212, 319488, 90.36000061035156], [1703932820.6812148, 327680, 91.41999816894531], [1703932853.263142, 335872, 89.80000305175781], [1703932885.8718483, 344064, 96.9800033569336], [1703932918.4178529, 352256, 99.69000244140625], [1703932951.0349298, 360448, 92.62000274658203], [1703932983.5163267, 368640, 97.48999786376953], [1703933016.1986384, 376832, 105.4800033569336], [1703933048.7888098, 385024, 101.5999984741211], [1703933081.4030085, 393216, 103.55000305175781], [1703933113.9441388, 401408, 98.58999633789062]]
# Extracting x and y values from the data
x = [entry[1] for entry in data]  # Assuming the second element is the step or epoch number
y = [entry[2] for entry in data]  # Assuming the third element is the ep_len mean

# Plotting the data
plt.plot(x, y)
plt.xlabel('Step or Epoch Number')
plt.ylabel('ep_len mean')
plt.title('Episode len Mean over Training Steps or Epochs')
plt.grid(True)
plt.show()

