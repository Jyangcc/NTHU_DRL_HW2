import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import gym
from collections import deque

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


class Agent():
    def __init__(self):
        self.model = DQN()
        checkpoint = torch.load('110062301_hw2_data.py', map_location=torch.device('cpu') )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.transforms1 = T.Compose( [T.ToTensor(), T.Grayscale()] )
        self.transforms2 = T.Compose( [T.Resize((84, 84), antialias=True), T.Normalize(0, 255)] )
        
        self.stack_state = deque(maxlen=4)
        self.frame_skip = 0
        self.step = 0
    
    def process(self, observation):
        observation = self.transforms1(observation.astype('int64').copy())
        observation = self.transforms2(observation.float()).squeeze(0)
        
        while len(self.stack_state) < 4:
            self.stack_state.append(observation)
        self.stack_state.append(observation)
        observation = torch.stack(list(self.stack_state)).unsqueeze(0)

        return observation
        
    
    def act(self, observation):
        if self.frame_skip % 4 == 0:
            observation = self.process(observation)
            self.last_action = self.model(observation).max(1)[1].view(1, 1).item()
        
        self.frame_skip +=1
        self.step += 1
        # print(self.step)
        if self.step == 2662:
            self.frame_skip = 0
            self.step = 0
            self.stack_state.clear()
        
        return self.last_action
