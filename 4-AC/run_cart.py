import gym
from A2C_agent import A2C
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

if __name__ == "__main__":
    #获取环境配置信息
    env = gym.make('CartPole-v0')
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    #设定参数
    learning_rate = 0.001
    reward_decay = 0.9
    #创建一个agent
    rl = A2C(state_dim, action_dim, learning_rate, reward_decay)
    #开始交互
    episode = 1000
    reward = np.zeros(episode) #存储每一个episode的奖励
    for ep in range(episode):
        total_reward = 0
        s = env.reset() #环境初始化
        print('--------This is the', ep,'episode----------')
        while True:
            a = rl.choose_action(s) #选动作
            s_, r, done, info = env.step(a) #执行动作
            rl.learn(s, a, r, s_, done)
            total_reward += r
            if done: #交互终止--game over
                print('the total reward:', total_reward)
                reward[ep] = total_reward
                break
            s = s_    #进入下一个状态      
    plt.plot(np.arange(episode), reward)
    plt.savefig('./reward_cartpole.png')
           