import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

gamma = 0.99
render = True
seed = 1
log_interval = 10

env = gym.make('CartPole-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.output = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.output(x), dim=1)
        return action_prob

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value

class PPOAgent():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self):
        super(PPOAgent,self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('../exp')

        self.actor_net_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(),3e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        #不会对weight求导
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        #根据概率采样一个动作
        action = c.sample()
        # 第一项tensor数组，第二项一个元素的tensor数组，第三项0或者1 ， 第四项0~1之间的小数。
        # print('action_prob:' + str(action_prob) +
        #       '----- action:'+str(action)+
        #       '-----  action.item:' +str(action.item()) +
        #       '-----  action_prob:' + str(action_prob[:,action.item()].item()) )
        #返回采样到动作 和模型预测出的动作被选择的概率
        return action.item(),action_prob[:,action.item()].item()

    def get_value(self,state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        #返回状态对应的值
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(),'../param/net_param/actor_net' + str(time.time())[:10], + '.pkl')
        torch.save(self.critic_net.state_dict(),'../param/net_param/critic_net' + str(time.time())[:10],+ '.pkl')

    def store_transition(self,transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self,i_ep):

        #buffer中的数据，buffer的item是Transition项
        #state 时间步t时的状态
        state = torch.tensor([t.state for t in self.buffer],dtype=torch.float)
        #是整轮游戏中每一个时间步的reward
        reward = [t.reward for t in self.buffer]
        #view相当于reshape,第一个是列数，第二个是行数，列数不确定时用-1表示
        action = torch.tensor([t.action for t in self.buffer],dtype = torch.long).view(-1,1)
        #这个和上一个就是select_action的返回值
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer],dtype = torch.float).view(-1,1)
        # print( 'action: ' + str(action))
        # print('old_action_log_prob: ' + str(old_action_log_prob))

        R = 0
        #一轮中每一个时间步t的长期收益(gains),也就是累积回报，从最后一步开始计算
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0,R)
        Gt = torch.tensor(Gt,dtype=torch.float)
        for i in range(self.ppo_update_time):
            for index_array in BatchSampler(SubsetRandomSampler(range(len(self.buffer))),self.batch_size,False):
                #print('index_array: '+str(index_array))
                if self.training_step % 1000 == 0:
                    print('I_ep {}, train {} times'.format(i_ep,self.training_step))
                #index数组项的长期收益
                Gt_index_array = Gt[index_array].view(-1,1)
                #数组中，每一步状态对应的值
                V = self.critic_net(state[index_array])
                delta = Gt_index_array - V
                #print('delta: ' + str(delta))
                #优势函数，不更新参数
                advantage = delta.detach()
                #print('advantage: ' + str(advantage))
                #epoch iteration, PPO core! gather:沿给定轴dim，将输入索引张量index指定位置的值进行聚合
                action_prob = self.actor_net(state[index_array]).gather(1,action[index_array]) # new policy

                #比例：新预测概率除以老预测动作概率
                ratio = (action_prob/old_action_log_prob[index_array])
                surr1 = ratio * advantage
                #将ratio范围限制到某个区间[1 - self.clip_param, self.clip_param]
                surr2 = torch.clamp(ratio,1 - self.clip_param,1 + self.clip_param) * advantage

                #update actor network
                #torch.min() surr1 surr2逐个元素相应位置对比，返回最小值到输出张量
                # mean 返回输入张量所有元素的均值, 也就是返回的是一个数
                action_loss = -torch.min(surr1,surr2).mean()
                #print('action_loss: ' + str(action_loss))
                self.writer.add_scalar('loss/action_loss',action_loss,global_step = self.training_step)
                self.actor_net_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(),self.max_grad_norm)
                self.actor_net_optimizer.step()

                #update critic network, 直接拿真值和预测值做均方误差来更新，真值的长期收益可以直接计算，预测值也是直接得到的
                value_loss = F.mse_loss(Gt_index_array,V)
                self.writer.add_scalar('loss/value_loss',value_loss,global_step = self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(),self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1
        #是不是代表了on-policy
        del self.buffer[:]

def play():
    agent = PPOAgent()
    for i_epoch in range(1000):
        state = env.reset()
        if render:env.render()

        for t in count():
            action,action_prob = agent.select_action(state)
            #根据当前状态和一个动作，得到下一个状态和这个动作的reward
            next_state,reward,done,_ = env.step(action)
            #只有状态是下一时间步的
            trans = Transition(state,action,action_prob,reward,next_state)
            if render: env.render()
            agent.store_transition(trans)
            state = next_state
            #本轮玩完时
            if done:
                #buffer中填的数据够一个batch后开始训练
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                agent.writer.add_scalar('liveTime/livestep',t,global_step=i_epoch)
                break

if __name__ == '__main__':
    play()
    print("end")

