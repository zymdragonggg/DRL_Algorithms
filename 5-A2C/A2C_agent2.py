'''
This is an A2C algorithm which uses critic network itself to obtain the next value of state.
action: discrete
experiment results: good performance
'''
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp

class MyNN(Model): 
    def __init__(self, s_dim, a_dim, hidden_nodes1, hidden_nodes2):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(hidden_nodes1, "relu")
        self.d2 = tf.keras.layers.Dense(hidden_nodes2, "relu")
        self.d3 = tf.keras.layers.Dense(a_dim, None)
        self(tf.keras.Input(shape = s_dim))

    def call(self, s):
        x = self.d1(s)
        x = self.d2(x)
        value = self.d3(x)
        return value

class A2C:
    def __init__(self, state_dim, action_dim, learning_rate = 0.001, reward_decay = 0.9):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = reward_decay
        self.num_hidden1 = 32
        self.num_hidden2 = 32

        self.actor_net = MyNN(self.s_dim, self.a_dim, self.num_hidden1, self.num_hidden2)
        self.critic_net = MyNN(self.s_dim, 1, self.num_hidden1, self.num_hidden2)
        self.optimizer1 = tf.keras.optimizers.Adam(self.lr)
        self.optimizer2 = tf.keras.optimizers.Adam(self.lr)
    
    def choose_action(self, s):
        s = s.reshape([1, self.s_dim]) #对传过来的数据升维度（[2,]-->[1,2]）
        action_prob = tf.nn.log_softmax(self.actor_net(s)) #将状态输入到网络中获得每个动作对应的log概率
        norm_dist = tfp.distributions.Categorical(probs = tf.math.exp(action_prob)) #处理为概率分布
        action = int(norm_dist.sample()) #这一步和上一步用来根据概率选动作，输出为tensor需要变成int
        return action

    def learn(self, s, a, r, s_, done):#一步一学习
        s, s_ = s.reshape([1, self.s_dim]), s_.reshape([1, self.s_dim])
        with tf.GradientTape(persistent = True) as tape:
            state_v = self.critic_net(s)
            state_v_ = self.critic_net(s_)
            td_error =  r + (1 - done) * self.gamma * state_v_ - state_v
            loss = tf.square(td_error) 
            
            all_action_prob = tf.nn.log_softmax(self.actor_net(s))
            norm_dist = tfp.distributions.Categorical(probs = tf.math.exp(all_action_prob))
            a_prob = norm_dist.log_prob(a)
            exp_v = - tf.reduce_mean(a_prob * td_error)
        loss_grads = tape.gradient(loss, self.critic_net.trainable_variables)
        grads = tape.gradient(exp_v, self.actor_net.trainable_variables)
        self.optimizer1.apply_gradients(zip(loss_grads, self.critic_net.trainable_variables))
        self.optimizer2.apply_gradients(zip(grads, self.actor_net.trainable_variables))
   
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
    plt.savefig('./reward_cart2.png')
