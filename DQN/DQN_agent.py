import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

class MyNN(Model): #神经网络搭建
    def __init__(self, s_dim, a_dim, hidden_nodes1, hidden_nodes2):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(hidden_nodes1, 'relu') # w: [s_dim, hidden_nodes1], b: [hidden_nodes1, ]
        self.d2 = tf.keras.layers.Dense(hidden_nodes2, 'relu')
        self.d3 = tf.keras.layers.Dense(a_dim, None)  # Q值不需要激活函数， [-inf, inf]
        self(tf.keras.Input(shape = s_dim))   # 初始化参数 [B, s_dim]

    def call(self, s):
        x = self.d1(s)
        x = self.d2(x)
        q = self.d3(x)
        return q    # [B, A]

class DeepQNetwork:
    def __init__(
            self,
            action_dim, #动作纬度
            state_dim,  #状态纬度
            epsilon = 0.2, #epislon-greedy
            reward_decay = 0.99, #折扣因子
            memory_size = 100000,  #经验池大小
            replace_target_iter = 1000, #目标网络参数更新间隔
            batch_size = 256, #经验池采样批大小
            learning_rate = 0.01 #学习率
            ):
        self.a_dim = action_dim
        self.s_dim = state_dim
        self.epsilon = epsilon
        self.gamma = reward_decay
        self.memory_size = memory_size  
        self.memory_counter = 0
        self.memory_now_size = 0
        self.memory = np.zeros((self.memory_size, self.s_dim*2+3))  # s:S, s':S, a:1, r:1, done:1]

        #neural network
        self.num_hidden1 = 32
        self.num_hidden2 = 32
        self.q_net = MyNN(self.s_dim, self.a_dim, self.num_hidden1, self.num_hidden2) #online network
        self.q_targt_net = MyNN(self.s_dim, self.a_dim, self.num_hidden1, self.num_hidden2) #target network
        tf.group([t.assign(s) for t, s in zip(self.q_targt_net.weights,self.q_net.weights)]) #使得两个网络参数在一开始保持一致
        self.replace_target_iter = replace_target_iter
        self.train_step_counter = 0
        self.batch_size = batch_size
        self.lr = learning_rate  # 不算小
        self.optimizer = tf.keras.optimizers.Adam(self.lr) 

        #模型保存
        self.ck_point = tf.train.Checkpoint(policy = self.q_net)
        self.saver = tf.train.CheckpointManager(self.ck_point, directory='./models', max_to_keep=5, checkpoint_name='zym_model')

    def save_model(self, episode): #用于保存模型
        self.saver.save(checkpoint_number=episode)

    def restore_model(self, directory):#用于验证模型
        self.ck_point.restore(tf.train.latest_checkpoint(directory))
        tf.group([t.assign(s) for t, s in zip(self.q_targt_net.weights,self.q_net.weights)])         

    def choose_action(self, state):
        s = np.array(state).reshape((1,len(state))) # [B=1, S]
        action_value = self.q_net(s)    #得到神经网络的所有输出,即所有动作的价值    # Q=[B=1, A]
        action = np.argmax(action_value)   # 1,
        if np.random.uniform() < self.epsilon:  #探索
            action = np.random.randint(0, self.a_dim)
        return action

    def store_transition(self, s, a, r, done, s_):
        transition = np.hstack((s, a, r, done, s_)) # S, 1, 1, 1
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_now_size = min(self.memory_size, self.memory_now_size+1)
        self.memory_counter += 1
    
    def learn(self):
        if self.train_step_counter % self.replace_target_iter == 0: #target网络参数替换
            tf.group([t.assign(s) for t, s in zip(self.q_targt_net.weights,self.q_net.weights)])
        if self.memory_now_size <= self.batch_size:  #如果batch_size大于现有经验
            sample_index = np.arange(self.memory_now_size) #全取
        else:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size) #随机采样经验
        batch_memory = self.memory[sample_index, :]#将采样到的经验放到batch_memory中

        mem_s = batch_memory[:, :self.s_dim]     #当前状态
        mem_s_= batch_memory[:, -self.s_dim:]    #下一个状态
        mem_action = batch_memory[:, self.s_dim].astype(int)  #获取到所有动作
        mem_reward = batch_memory[:, self.s_dim + 1]    #获取到所有奖励
        mem_done = batch_memory[:, self.s_dim + 2]    #获取到所有奖励
        q_loss = self.train(mem_s, mem_s_, mem_action, mem_reward, mem_done)

    def train(self, s, s_, a, r, done):
        '''
        s: [B, S]
        s_: [B, S]
        a: [B, ]
        r: [B, ]
        done: [B, ]
        '''
        with tf.GradientTape() as tape:
            q_eval_current = self.q_net(s)  #获得当前状态下所有动作的估计值 [B, A]
            mem_action_onehot = tf.one_hot(a, self.a_dim) #对经验池动作处理成one hot [B, ] => [B, A]
            q_eval = tf.reduce_sum(tf.multiply(q_eval_current, mem_action_onehot), axis=1)#计算出估计值q(s,a) [B, A] => [B, ]

            q_target_next = self.q_targt_net(s_) #获得下一个状态的所有动作目标值    # [B, A]            
            q_target = r + (1 - done) * self.gamma * np.max(q_target_next, axis=1) #计算出target值q(s_,a)=r+gamma*q(s_,a_)  [B, ]
            
            td_error = q_eval - q_target #q(s, a) - r + gamma * q(s_, a_)
            q_loss = tf.reduce_mean(tf.square(td_error))  # [B, ] => 1,
        grads = tape.gradient(q_loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        self.train_step_counter += 1
        return q_loss