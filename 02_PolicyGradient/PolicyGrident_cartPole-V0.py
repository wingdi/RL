# 蒙特卡罗策略梯度
# 数据池：保存没一轮的运行数据，这次训练之后就清空
# 累积回报：累积后面每一步的回报*gamma
# 网络：

import numpy as np
import gym
import matplotlib.pyplot as plt
from itertools import count
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

#超参数
learning_rate = 0.01
gamma = 0.98 #折扣因子

#参数
num_episode = 5000
batch_size = 32

#env
env = gym.make('CartPole-v0')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

class policy_net(nn.Module):

    def __init__(self,env):
        super(policy_net, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.softmax(self.fc2(x),dim=-1)
        return x

pNetwork = policy_net(env)


#绘图布局
def plot(episode_durations):
    plt.ion()
    plt.figure(2)
    plt.clf() #清除所有轴，但是窗口打开，这样它可以被重复使用
    duration_t = torch.FloatTensor(episode_durations)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(duration_t.numpy())

    if len(duration_t) >= 100:
        means = duration_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())
    plt.pause(0.00001)

class PGAgent():

    def __init__(self):
        super(PGAgent,self).__init__()
        self.network = pNetwork
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr = learning_rate)

    #更新network的参数
    def train(self):
        episode_durations = []
        # batch_history 用来保存每一轮的状态、动作和奖励
        state_pool = []
        action_pool = []
        reward_pool = []
        steps = 0

        for episode in range(num_episode):
            state = env.reset()
            state = torch.from_numpy(state).float()
            state = Variable(state)
            env.render()

            #训练1轮，并把数据保存到数据池
            for t in count():
                #预测各个动作被选择的概率
                probs = self.network(state)
                c = Categorical(probs)
                #采样一个动作
                action = c.sample()
                action = action.data.numpy().astype('int32')
                #根据预测的动作获取 下一步的状态和奖励
                next_state, reward, done, info = env.step(action)
                reward = 0 if done else reward  # correct the reward
                env.render()

                #把state、action和相应的奖励放到数据池中
                state_pool.append(state)
                action_pool.append(float(action))
                reward_pool.append(reward)

                #状态赋新值
                state = next_state
                state = torch.from_numpy(state).float()
                state = Variable(state)

                steps += 1

                if done:
                    episode_durations.append(t + 1)
                    plot(episode_durations)
                    break

            # episode每隔一个batch_size就更新策略，并且更新参数θ,每隔32轮才更新一次θ
            if episode > 0 and episode % batch_size == 0:
                r = 0
                #range(5) = [0,1,2,3,4]; i是reward中的每个值，step ：没一轮走的步数。
                # 计算这一轮中，每一步的真实回报。并且赋给reward_pool[i]
                for i in reversed(range(steps)):
                    if reward_pool[i] == 0:
                        r = 0
                    else:
                        r = reward_pool[i] + r * gamma
                        #奖励叠加得到回报
                        reward_pool[i] = r

                # Normalize reward
                reward_mean = np.mean(reward_pool) #均值
                rewad_std = np.std(reward_pool) #标准差
                reward_pool = (reward_pool - reward_mean) / rewad_std #归一化，处理后数据符合标准正态分布

                # gradiend desent
                #将参数的梯度置为零，也就是把loss关于weight的导数置为0
                pgAgent.optimizer.zero_grad()
                for i in range(steps):
                    state = state_pool[i]
                    action = Variable(torch.FloatTensor([action_pool[i]]))
                    reward = reward_pool[i]

                    probs = self.network(state)
                    c = Categorical(probs)

                    #负号：因为优化器是梯度下降，而我们要做的是梯度上升。
                    # c.log_prob(action)返回在action处的概率密度函数的对数
                    # 也就是在s状态下，执行action动作的概率的对数
                    loss = -c.log_prob(action) * reward
                    loss.backward()

                self.optimizer.step()

                # clear the batch pool
                state_pool = []
                action_pool = []
                reward_pool = []
                steps = 0

if __name__ == '__main__':
    pgAgent = PGAgent()
    pgAgent.train()