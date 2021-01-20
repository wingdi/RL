
# base on https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
# 数据池：不分轮次，不清空，超出容量时从头覆盖。
# 输入值：两个网络输入值 都是从数据池中随机采样时间步
# 累积回报：是目标网络的输出
# 网络：有两个，一个是估计网络， 一个是目标网络，目标网络的参数从估计网络复制而来

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

#超参数
learning_rate = 0.01  #学习率
gamma = 0.90  #折扣因子
epsilon = 0.9 #探索率

batch_size = 128
memory_pool_size = 2000 #memory存储容量
step_interval = 100 #每隔100步存储到memory一次

#env
env = gym.make("CartPole-v0")
action_space = env.action_space.n
state_space = env.observation_space.shape[0]

class shared_net(nn.Module):
    def __init__(self):
        super(shared_net,self).__init__()
        self.fc1 = nn.Linear(state_space, 50)
        self.fc2 = nn.Linear(50, 30)
        self.out = nn.Linear(30, action_space)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQNAgent():
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.eval_net, self.target_net = shared_net(),shared_net()
        self.init_memory_pool()  # 用来装训练数据
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def init_memory_pool(self):
        #记忆池与轮数无关，数据多于记忆池容量时就从头赋值。训练用的数据也是采样独立的时间步
        self.memory_pool = np.zeros((memory_pool_size, state_space * 2 + 2))  # 2000 x 10
        self.learn_step_counter = 0
        self.memory_counter = 0

    def train(self):
        #每隔100步更新一次target_net的参数
        if self.learn_step_counter % step_interval == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        #sample batch from memory
        sample_index = np.random.choice(memory_pool_size, batch_size)
        #随机采样的时间步都是独立的
        #print('sample_index:  ' + str(sample_index))
        batch_memory = self.memory_pool[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :state_space])
        batch_action = torch.LongTensor(batch_memory[:, state_space:state_space + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, state_space + 1:state_space + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -state_space:])

        #估计值
        q_eval = self.eval_net(batch_state).gather(1,batch_action)
        #得到下一个状态对应的回报
        q_next = self.target_net(batch_next_state).detach()
        #目标值，也就是长期收益，后半部分由目标网络直接输出,采用的贪心算法
        q_target = batch_reward + gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.loss_func(q_eval,q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    #根据state获取两个Action的Value，然后选择Value最大的那个动作
    def choose_action(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state),0)
        if np.random.randn()<=epsilon: # 贪心策略
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value,1)[1].data.numpy()
            action = action[0]
        else:#随机选择
            action = np.random.randint(0, action_space)
        return action

    def store_memory(self,state,action,reward,next_state):
        temp = np.hstack((state,[action,reward],next_state))
        #当超过2000时，就从头替换赋值了
        index = self.memory_counter % memory_pool_size
        self.memory_pool[index, :] = temp
        self.memory_counter += 1

    # 奖励函数 x小车的位置，theta 木棒的角度，每一步价值的设计
    def get_reward(self,state):
        x, x_dot, theta, theta_dot = state
        #x_threshold 最大边距
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return reward

def plot(reward_list):
    plt.ion()
    plt.figure(2)
    plt.clf()  # 清除所有轴，但是窗口打开，这样它可以被重复使用
    duration_t = torch.FloatTensor(reward_list)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(duration_t.numpy())

    if len(duration_t) >= 100:
        means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.00001)


def play_step():
    dqn = DQNAgent()
    episodes = 400
    reward_list = []
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        step_counter = 0
        while True:
            step_counter += 1
            env.render()
            action = dqn.choose_action(state)
            next_state,_,done,info = env.step(action)
            reward = dqn.get_reward(next_state)
            #把玩的数据存到memory_pool里
            dqn.store_memory(state,action,reward,next_state)
            ep_reward += reward
            #当数据池存满后，开始从里面随机取样进行训练
            if dqn.memory_counter >= memory_pool_size:
                dqn.train()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                reward_list.append(ep_reward)
                break
            state = next_state
        # 走够200步就胜利，env环境封装的最多就只能走200步，参考env.unwrapped
        if step_counter >= 200:
            print('you are the winner!')
            break
        plot(reward_list)

if __name__ == '__main__':
    play_step()