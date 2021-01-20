# base on https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
import gym, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

learning_rate = 0.01
gamma = 0.99
episodes = 20000
render = True
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction',['log_prob','value'])

#env
env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)
torch.manual_seed(1)
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

class actor_network(nn.Module):
    def __init__(self):
        super(actor_network,self).__init__()
        self.fc1 = nn.Linear(state_space,32)
        self.output = nn.Linear(32,action_space)
        self.save_actions = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        os.makedirs('./Ac_CartPole-v0', exist_ok=True)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        action_score = self.output(x)
        return F.softmax(action_score,dim=0)

    def select_action(self,state):
        state = torch.from_numpy(state).float()
        probs = actor(state)
        state_value = critic(state)
        m = Categorical(probs)
        action = m.sample()
        actor.save_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

class critic_network(nn.Module):
    def __init__(self):
        super(critic_network,self).__init__()
        self.fc1 = nn.Linear(state_space,32)
        self.output = nn.Linear(32,1)
        self.rewards = []
        self.optimizer = optim.Adam(self.parameters(),lr=learning_rate)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        state_value = self.output(x)
        return state_value

actor = actor_network()
critic = critic_network()

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(steps)
    RunTime = len(steps)

    path = './Ac_CartPole-v0' + 'RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 200 ==0:
        plt.savefig(path)
    plt.pause(0.0000001)

def actor_critic_train():
    R = 0
    save_actions = actor.save_actions
    policy_loss = []
    value_loss = []
    rewards = []

    for r in critic.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0,R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean())/(rewards.std() + eps)

    for(log_prob,value),r in zip(save_actions,rewards):
        value_loss.append(F.smooth_l1_loss(value,torch.tensor([r]))) # todo
        reward = r - value.item()
        policy_loss.append(-log_prob * reward)
    actor.optimizer.zero_grad()
    critic.optimizer.zero_grad()
    # loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    actor_loss = torch.stack(policy_loss).sum()
    # stack 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
    critic_loss = torch.stack(value_loss).sum()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    actor.optimizer.step()
    critic.optimizer.step()

    del critic.rewards[:]
    del actor.save_actions[:]

def main():
    running_reward = 10
    live_time = []
    for i_episode in count(episodes):
        state = env.reset()
        for t in count():
            action = actor.select_action(state)
            state,reward,done,info = env.step(action)
            if render:env.render()
            critic.rewards.append(reward)
            if done or t >= 1000:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        live_time.append(t)
        plot(live_time)
        if i_episode % 100 == 0:
            modelPath = './Ac_CartPole-v0/ModelTraing'+str(i_episode)+'Times.pkl'
            torch.save(actor, modelPath)
        actor_critic_train()

if __name__ == '__main__':
    main()