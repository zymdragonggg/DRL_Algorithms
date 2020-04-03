import gym
from PG_agent import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    #获取环境配置信息
    env = gym.make('CartPole-v0')
    env.seed(3)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    #设定参数
    learning_rate = 0.01
    reward_decay = 0.9
    #创建一个agent
    RL = PolicyGradient(action_dim, state_dim, learning_rate, reward_decay)
    #开始交互
    episode = 500
    reward = np.zeros(episode) #存储每一个episode的奖励
    for ep in range(episode):
        if ep % 50 == 0:
            RL.save_model(ep)
        s = env.reset() #环境初始化
        print('--------This is the', episode,'episode----------')
        while True:
            a = RL.choose_action(s) #选动作
            s_, r, done, info = env.step(a) #执行动作
            RL.store_transition(s, a, r) #存储每一步交互
            if done: #交互终止--game over
                print('the total reward:', sum(RL.ep_r))
                reward[ep] = sum(RL.ep_r)
                break
            s = s_    #进入下一个状态      
        RL.learn() #agent学习

    plt.plot(np.arange(episode),reward)
    plt.savefig('./reward.png')   
           