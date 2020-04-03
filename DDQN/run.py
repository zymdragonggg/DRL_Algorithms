import gym
from DDQN_agent import DoubleDeepQNetwork
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    #获取环境配置信息
    env = gym.make('CartPole-v0')
    env.seed(3)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    #创建一个agent
    RL = DoubleDeepQNetwork(action_dim, state_dim)
    #开始交互
    episode = 7000
    reward = np.zeros(episode) #存储每一个episode的奖励
    for ep in range(episode):
        if ep % 1000 == 0:
            RL.save_model(ep)
        total_reward = 0
        s = env.reset()
        print('--------This is the', ep ,'episode----------')
        while True:
            a = RL.choose_action(s)
            s_, r, done, info = env.step(a)
            RL.store_transition(s, a, r, done, s_)
            RL.learn()
            total_reward += r
            if done:
                print('the total reward:', total_reward )
                reward[ep] = total_reward
                break
            s = s_
    plt.plot(np.arange(episode), reward)
    plt.savefig('./reward.png')
           