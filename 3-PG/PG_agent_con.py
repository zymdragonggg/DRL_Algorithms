import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp

class MyNN(Model):
    def __init__(self, s_dim, a_dim, hidden_nodes1, hidden_nodes2):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(hidden_nodes1, 'relu') 
        self.d2 = tf.keras.layers.Dense(hidden_nodes2, 'relu')
        self.mu = tf.keras.layers.Dense(a_dim, "tanh")  #[-1, 1]
        self.sigma = tf.keras.layers.Dense(a_dim, tf.nn.sigmoid) #[0, 1]
        self(tf.keras.Input(shape = s_dim))   

    def call(self, s):
        x = self.d1(s)
        x = self.d2(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma    

class PolicyGradient:
    def __init__(
            self,
            action_dim,
            state_dim,
            learning_rate = 0.01,
            reward_decay = 0.95,
    ):
        self.a_dim = action_dim
        self.s_dim = state_dim
        self.lr = learning_rate
        self.gamma = reward_decay
        self.num_hidden1 = 32
        self.num_hidden2 = 32

        self.ep_s, self.ep_a, self.ep_r, self.ep_d = [], [], [], []
        self.net = MyNN(self.s_dim, self.a_dim, self.num_hidden1, self.num_hidden2)  
        self.optimizer = tf.keras.optimizers.Adam(self.lr) 
        self.ck_point = tf.train.Checkpoint(policy = self.net)
        self.saver = tf.train.CheckpointManager(self.ck_point, directory='./models_Con', max_to_keep=5, checkpoint_name='zym_')

    def save_model(self, episode):
        self.saver.save(checkpoint_number=episode)

    def restore_model(self, directory):
        self.ck_point.restore(tf.train.latest_checkpoint(directory))          

    def choose_action(self, s):
        s = s.reshape([1, self.s_dim]) #对传过来的数据升维度（[2,]-->[1,2]）
        mu, sigma = self.net(s) #将状态输入到网络中获得每个动作对应高斯分布的参数
        norm_dist = tfp.distributions.Normal(mu, sigma)  #生成动作的高斯分布
        action = norm_dist.sample() #根据分布选择的动作
        action = tf.clip_by_value(action, -1.0, 1.0) #对结果进行截断，防止取到非范围内的动作值
        action = action.numpy().flatten() #对结果进行处理，变成环境可以理解的动作
        return action

    def store_transition(self, s, a, r, done):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_d.append(done)

    def learn(self):
        discounted_ep_r = self._discount_and_norm_rewards()
        s = np.array(self.ep_s)
        a = np.array(self.ep_a)
        with tf.GradientTape() as tape:
            mu, sigma = self.net(s)
            norm_dist = tfp.distributions.Normal(mu, sigma)
            sa_prob = norm_dist.log_prob(a)
            sa_prob_mean = tf.reduce_mean(sa_prob, axis = -1) #先对所有动作求平均
            loss = -tf.reduce_mean(sa_prob_mean * discounted_ep_r)  
        loss_grads = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(loss_grads, self.net.trainable_variables))
        self.ep_s, self.ep_a, self.ep_r, self.ep_d = [], [], [], [] 

    def _discount_and_norm_rewards(self): # 
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_r)
        running_add = 0
        for t in reversed(range(0, len(self.ep_r))):
            running_add = running_add * self.gamma * (1 - self.ep_d[t]) + self.ep_r[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs   
    