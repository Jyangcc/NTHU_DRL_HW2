import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np

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

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.model = DQN()
        self.model.load_state_dict(torch.load('./110062301_hw2_data.py'), map_location=torch.device('cpu'))
        
        self.transform = T.Compose(
            [T.ToTensor(), T.Grayscale(), T.Resize((84,84), antialias=True), T.Normalize(0, 255) ]
        )
        
        self.skip_frame = 0
        self.last_action = 0
        
        self.prev_state = torch.zeros(1,4,84,84)
        
        

    def act(self, observation):
        
        if self.skip_frame % 4 == 0:
            observation = self.transform(observation)
            observation = torch.tensor(np.array([observation], copy=False))
            self.prev_state = torch.cat((self.prev_state[0][1:], observation)).reshape(1,4,84,84)
            
            
            action = self.model(self.prev_state).max(1)[1].view(1, 1)
            self.last_action = action.item()
            
        self.skip_frame += 1
        
        return self.last_action
