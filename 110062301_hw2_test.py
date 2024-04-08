import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt



import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from collections import deque

import numpy as np
import os, time

################################
###     Deep Q Network       ###
################################

class DQN(nn.Module):
    def __init__(self, in_channels = 4, num_actions=12):

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        
        self.flatten = nn.Flatten()
        
        self.Value1 = nn.Linear(3136, 256)
        self.Value2 = nn.Linear(256, 1)
        
        self.Adv1 = nn.Linear(3136, 256)
        self.Adv2 = nn.Linear(256, num_actions)
        
        torch.nn.init.normal_(self.Value1.weight, 0.0, 0.01)
        torch.nn.init.normal_(self.Value2.weight, 0.0, 0.01)
        torch.nn.init.normal_(self.Adv1.weight, 0.0, 0.01)
        torch.nn.init.normal_(self.Adv2.weight, 0.0, 0.01)
        
    def forward(self, x):
        
        # Convolutional Layers
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.flatten(x)
        
        # Dueling Network State Value
        value = F.elu(self.Value1(x))
        value = self.Value2(value)
        
        
        # Dueling Network Advantage Value
        adv = F.elu(self.Adv1(x))
        adv = self.Adv2(adv)
        
        
        mean_adv = torch.mean(adv, dim=1, keepdim=True)
        Rev = value + adv - mean_adv
        
        return Rev



################################
###          Agent           ###
################################

# class Agent(object):
#     """Agent that acts randomly."""
#     def __init__(self):
#         self.model = DQN()
#         checkpoint = torch.load('110062301_hw2_data.py', map_location=torch.device('cpu') )

#         self.model.load_state_dict(checkpoint['model_state_dict'])
        
#         self.transform1 = T.Compose(
#             [T.ToTensor(), T.Grayscale() ]
#         )
        
#         self.transform2 = T.Compose( [T.Resize((84,84), antialias=True), T.Normalize(0, 255) ])
        
#         self.skip_frame = 0
#         self.last_action = 0
        
#         self.prev_state = torch.rand(1,4,84,84)
#         print("init")
        
        

#     def act(self, observation):
        
#         if self.skip_frame % 4 == 0:
#             # observation = np.transpose(observation, (2, 0, 1))
#             print(observation.shape)


#             observation = self.transform1(observation.copy())
#             print(observation.shape)
#             observation = self.transform2(observation.float())
#             print(observation.shape)
#             print("===")
#             self.prev_state = torch.cat((self.prev_state[0][1:], observation)).reshape(1,4,84,84)
            
            
#             action = self.model(self.prev_state).max(1)[1].view(1, 1)
#             self.last_action = action.item()
            
#         self.skip_frame += 1
        
#         return self.last_action



class Agent():
    def __init__(self):
        self.model = DQN()
        
        checkpoint = torch.load('110062301_hw2_data.py', map_location=torch.device('cpu') )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        
        self.action_space = [i for i in range(12)]
        self.transforms1 = T.Compose(
            [T.ToTensor(), T.Grayscale()]
        )
        self.transforms2 = T.Compose(
            [T.Resize((84, 84), antialias=True), T.Normalize(0, 255)]
        )
        self.frames = deque(maxlen=4)
        self.frame_skip = 0
        
    
    def act(self, observation):
        if self.frame_skip % 4 == 0:
            observation = self.transforms1(observation.astype('int64').copy())
            observation = self.transforms2(observation.float()).squeeze(0)
            while len(self.frames) < 4:
                self.frames.append(observation)
                
            self.frames.append(observation)
            observation = gym.wrappers.frame_stack.LazyFrames(list(self.frames))
            observation = observation[0].__array__() if isinstance(observation, tuple) else observation.__array__()
            observation = torch.tensor(observation).unsqueeze(0)
            self.last_action = self.model(observation).max(1)[1].view(1, 1).item()
            
            
        self.frame_skip += 1
        return self.last_action





# if __name__ == '__main__': 
#     # Create Environment
#     env = gym_super_mario_bros.make('SuperMarioBros-v0')
#     env = JoypadSpace(env, COMPLEX_MOVEMENT)

#     agent = Agent()
#     tot_reward = 0

#     for i in range(10):
#         r = 0
#         done = False
#         state = env.reset()
#         start_time = time.time()

#         while not done:
#             action = agent.act(state)
#             next_state, reward, done, info = env.step(action)
            
#             # env.render()

#             if time.time() - start_time > 120:
#                 break

#             tot_reward += reward
#             r += reward
#             state = next_state
#             # env.render('human')
#         print(f'Game #{i}: {r}')
#         print(f'====================')
#         time.sleep(10)


#     env.close()
#     print(f'mean_reward: {tot_reward/10}')
