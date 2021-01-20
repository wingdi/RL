import argparse
import pickle
from collections import namedtuple
from itertools import count

import os
import numpy as np

import gym
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import grad
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

device = 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument("--env_name",default="Pendulum-v0")
parser.add_argument('--tau',default=0.005,type=float) #目标平滑系数
parser.add_argument('--target_update_interval',default=1,type=int)
parser.add_argument('--gradient_steps',default=1,type=int)

parser.add_argument('--learning_rate',default=3e-4,type=int)
parser.add_argument('--gamma',default=0.99,type=int)
parser.add_argument('--capacity',default=1000,type=int)
parser.add_argument('--iteration',default=100000,type=int) #玩游戏的局数
parser.add_argument('--batch_size',default=128,type=int)
parser.add_argument('--seed',default=1,type=int)

# 可选参数
parser.add_argument('--num_hidden_layers',default=2,type=int)
parser.add_argument('--num_hidden_units_per_layer',default=256,type=int)
parser.add_argument('--sample_frequency',default=256,type=int)
parser.add_argument('--activation',default='Relu',type=str)
parser.add_argument('--render',default=True,type=bool)
parser.add_argument('--log_interval',default=2000,type=int)
parser.add_argument('--load',default=False,type=bool) # load model
args = parser.parse_args()

"""
log 就是对数
mu 就是正太分布中期望
log_std 就是正态分布中的标准差 
log_prob 就是网络输出的预测值取对数
sigmma 也是标准差
dtype 是data type
"""

#action归一化 也就是重写了gym中的action
class NormalizedActions(gym.ActionWrapper):

    def action(self,action):
        low = self.action_space.low
        high = self.action_space.high
        action = low + (action + 1.0) * 0.5 * (high-low)
        action = np.clip(action,low,high)
        #print('action：' + str(action) + '  low:  ' + str(low) + ' high: ' + str(high))
        return action

    #好像没用到这个函数
    def reverse_action(self,action):
        low = self.action_space.low
        high = self.action_space.high
        action = 2*(action - low)/(high - low) - 1
        action = np.clip(action,low,high)
        #print('reverse_action： ' + str(action))
        return action

env = NormalizedActions(gym.make(args.env_name))

#Set seeds ,再调用random时生成的随机数是同一个
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
print('state_dim: ' + str(state_dim))
action_dim = env.action_space.shape[0]
print('action_dim:  ' + str(action_dim))
max_action = float(env.action_space.high[0])
#最小值，用来限制估计action的最小值？todo
min_Val = torch.tensor(1e-7).float()
Temp = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])

# 3 - 256 - 256 - 1
class actor_net(nn.Module):
    def __init__(self,state_dim,min_log_std=-20,max_log_std=2):
        super(actor_net, self).__init__()
        self.fc1 = nn.Linear(state_dim,256)
        self.fc2 = nn.Linear(256,256)
        self.mu_head = nn.Linear(256,1)
        self.log_std_head = nn.Linear(256,1)
        self.max_action = max_action

        #log 标准差
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head,self.min_log_std,self.max_log_std)
        return mu,log_std_head

# 3 - 256 - 256 - 1
class critic_net(nn.Module):
    def __init__(self,state_dim):
        super(critic_net, self).__init__()
        self.fc1 = nn.Linear(state_dim,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3 - 256 - 256 - 1
class Q_net(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim , 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,1)

    def forward(self,s,a):
        s = s.reshape(-1,state_dim)
        a = a.reshape(-1,action_dim)
        # concatenate 把s,a拼接在一期
        x = torch.cat((s,a),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class SAC():
    def __init__(self):
        super(SAC,self).__init__()
        # 四个网络
        self.policy_net = actor_net(state_dim).to(device)
        self.value_net = critic_net(state_dim).to(device)
        self.Q_net = Q_net(state_dim, action_dim).to(device)
        self.target_value_net = critic_net(state_dim).to(device)

        self.replay_buffer = [Temp] * args.capacity
        self.buffer_counter = 0  # 计数器
        # 三个优化器
        self.policy_optimizer = optim.Adam(self.value_net.parameters(),lr = args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(),lr = args.learning_rate)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(),lr = args.learning_rate)
        #训练次数
        self.training_num = 1
        #后期可以用tensorboard在网页中可视化数据
        self.writer = SummaryWriter('./exp-SAC')

        self.value_init_loss = nn.MSELoss()
        self.Q_init_loss = nn.MSELoss()
        # param 也就是模型中的weight, 目标网络 从value网络复制参数（但这是init方法）
        for target_param,param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/',exist_ok=True)

    #根据policy_net、state，对动作采样
    def select_action(self,state):
        state = torch.FloatTensor(state).to(device)
        mu,log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        #创建一个正态分布
        dist = Normal(mu,sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        # print('action:  ' + str(action))
        # print('action.item():  ' + str(action.item()))
        # 只有一个action 当值为0.17时表示用0.17的力向右推
        return action.item() # return a scalar, float32

    def store(self, s, a, r, next_state, done):
        # 避免溢出，来一个存一个，空间不够了从头覆盖
        i = self.buffer_counter % args.capacity
        temp = Temp(s, a, r, next_state, done)
        self.replay_buffer[i] = temp
        self.buffer_counter += 1

    # 获取action 预测值
    def get_action_log_prob(self,state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        # 得到分布
        dist = Normal(batch_mu,batch_sigma)
        # 128维 也就是batch_size,每步采样一个
        sample_result = dist.sample()
        #print('sample_result: ' + str(sample_result) + 'sample_sesult_shape: ' + str(sample_result.size()))
        # 将action 也就是力的大小 限制在-1 ~ 1
        action = torch.tanh(sample_result)
        # print('action: ' + str(action))
        # pow 2次方, 对预测值的动作结果取log ?
        log_prob = dist.log_prob(sample_result) - torch.log(1 - action.pow(2) + min_Val)
        print('log_prob: ' + str(log_prob))
        return action, log_prob, sample_result, batch_mu, batch_log_sigma

    def update(self):
        if self.training_num % 500 == 0:
            print('Training ... {}'.format(self.training_num))
        s = torch.tensor([t.s for t in self.replay_buffer]).float().to(device)
        a = torch.tensor([t.a for t in self.replay_buffer]).float().to(device)
        r = torch.tensor([t.r for t in self.replay_buffer]).float().to(device)
        next_state = torch.tensor([t.s_ for t in self.replay_buffer]).float().to(device)
        d = torch.tensor([t.d for t in self.replay_buffer]).float().to(device)

        for _ in range(args.gradient_steps):
            # 从数据池中随机取样
            index = np.random.choice(range(args.capacity),args.batch_size,replace=False)
            bn_s = s[index]
            bn_a = a[index].reshape(-1,1)
            bn_r = r[index].reshape(-1,1)
            b_next_state = next_state[index]
            bn_d = d[index].reshape(-1,1)

    #-----------------V_loss 推导过程 -------------------------------------------------------#
            value_net_outp = self.value_net(bn_s) #估计值
            sample_action,log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(bn_s)
            # 从策略网络采样得到动作 给到Q_2
            Q_2_output = self.Q_net(bn_s,sample_action)
            next_value = Q_2_output - log_prob
            # the actions are sampled according to the current policy,not replay buffer
            V_loss = self.value_init_loss(value_net_outp, next_value.detach()) # J_V
    # -----------------V_loss 推导过程 -------------------------------------------------------#

            #print('V_loss: ' + str(V_loss))
            #平均后没有变化，因为只有一个值
            V_loss = V_loss.mean()
            #print('V_loss_mean: ' + str(V_loss))

    # -----------------Q_loss 推导过程 -------------------------------------------------------#
            target_value_outp = self.target_value_net(b_next_state)
            next_Q_value = bn_r + (1 - bn_d) * args.gamma * target_value_outp
            Q_1_output = self.Q_net(bn_s, bn_a)
            #Single Q_net this is different from original paper Q这里和原论文不同
            Q_loss = self.Q_init_loss(Q_1_output, next_Q_value.detach()) # J_Q
            Q_loss = Q_loss.mean()
    # -----------------Q_loss 推导过程 -------------------------------------------------------#

    # -----------------policy_loss 推导过程 -------------------------------------------------------#
            log_policy_target = Q_2_output - value_net_outp
            pi_loss = log_prob * (log_prob - log_policy_target).detach()
            pi_loss = pi_loss.mean()
    # -----------------policy_loss 推导过程 -------------------------------------------------------#

            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.training_num)
            self.writer.add_scalar('Loss/Q_loss', Q_loss, global_step = self.training_num)
            self.writer.add_scalar('Loss/pi_loss', pi_loss, global_step = self.training_num)
            #mini batch gradient descent
            self.value_optimizer.zero_grad()
            #print('V_loss: ' + str(V_loss))
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(),0.5)
            self.value_optimizer.step()

            self.Q_optimizer.zero_grad()
            # 保留图=True
            Q_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net.parameters(),0.5)
            self.Q_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(),0.5)
            self.policy_optimizer.step()

            #soft update，目标权重参数重新赋值
            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - args.tau)+ param * args.tau)

            self.training_num += 1

    def save(self):
        torch.save(self.policy_net.state_dict(),'./SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(),'./SAC_model/value_net.pth')
        torch.save(self.Q_net.state_dict(),'./SAC_model/Q_net.pth')
        print('===============')
        print("Model has been saved")
        print("=======================")

    def load(self):
        torch.load(self.policy_net.state_dict(),'./SAC_model/policy_net.pth')
        torch.load(self.value_net.state_dict(),'./SAC_model/value_net.pth')
        torch.load(self.Q_net.state_dict(),'./SAC_model/Q_net.pth')
        print()

def main():
    agent = SAC()
    if args.load:agent.load()
    if args.render:
        env.reset()
        env.render()
    print('============')
    print('Collection Experience.. ')
    print('============')

    #每一轮的总得分
    ep_r = 0
    for i in range(args.iteration):
        state = env.reset()
        #每轮达到200步时还没有结束也没有胜利，就重新开始。
        #每轮达到200步时还没有结束也没有胜利，就重新开始。
        for t in range(200):
            action = agent.select_action(state)
            #根据当前state和action得到 reward和next_state
            next_state,reward,done,info = env.step(np.float32(action))
            ep_r += reward
            if args.render:
                env.render()
            # 每个step都存储
            agent.store(state,action,reward,next_state,done)
            #当存满的时候，也就是走10000步时，模型参数更新一次
            if done:
                print('episode_done: ' + str(t))
            if agent.buffer_counter >= args.capacity:
                agent.update()
            #把得到的state赋值给当前状态
            state = next_state
            #当这一轮游戏结束 或者达到了200步
            if done or t == 199:
                if i % 10 == 0:
                    print("Ep_i {}, the ep_r is {}, the t is {}".format(i, ep_r, t))
                break
            # 每迭代两千次就把模型保存到本地一次
        if i % args.log_interval == 0:
            agent.save()
        agent.writer.add_scalar('ep_r',ep_r, global_step=i)
        ep_r = 0
if __name__ == '__main__':
    main()




