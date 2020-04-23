import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp

class ActorNet(Model): #actor网络:连续动作输出动作的高斯分布
    def __init__(self, s_dim, a_dim, hidden_nodes1, hidden_nodes2):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(hidden_nodes1, "relu")
        self.d2 = tf.keras.layers.Dense(hidden_nodes2, "relu")
        self.mu = tf.keras.layers.Dense(a_dim, "tanh")  #均值
        self.sigma = tf.keras.layers.Dense(a_dim, tf.nn.sigmoid) #方差
        self(tf.keras.Input(shape = s_dim))

    def call(self, s):
        x = self.d1(s)
        x = self.d2(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

class CriticNet(Model): #critic网络：输出状态的价值
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

        self.actor_net = ActorNet(self.s_dim, self.a_dim, self.num_hidden1, self.num_hidden2)
        self.critic_net = CriticNet(self.s_dim, 1, self.num_hidden1, self.num_hidden2)
        self.optimizer1 = tf.keras.optimizers.Adam(self.lr)
        self.optimizer2 = tf.keras.optimizers.Adam(self.lr)
    
    def choose_action(self, s):
        s = s.reshape([1, self.s_dim]) #对传过来的数据升维度（[2,]-->[1,2]）
        mu, sigma = self.actor_net(s) #将状态输入到网络中获得每个动作对应高斯分布的参数
        norm_dist = tfp.distributions.Normal(mu, sigma)  #生成动作的高斯分布
        action = norm_dist.sample() #根据分布得到选择的动作
        action = tf.clip_by_value(action, -1.0, 1.0)  #选出动作
        action = action.numpy().flatten() #处理动作为环境可理解的
        return action

    def learn(self, s, a, r, s_, done):   #一步一学习
        s, s_ = s.reshape([1, self.s_dim]), s_.reshape([1, self.s_dim])
        with tf.GradientTape(persistent = True) as tape:
            state_v = self.critic_net(s)
            state_v_ = self.critic_net(s_)
            td_error =  r + (1 - done) * self.gamma * state_v_ - state_v #得到td_error
            critic_loss = tf.square(td_error)  #得到critic网络的误差
            
            mu, sigma = self.actor_net(s) 
            norm_dist = tfp.distributions.Normal(mu, sigma)
            a_prob = norm_dist.log_prob(a) #得到动作的概率
            exp_v = - tf.reduce_mean(a_prob * td_error) 
        critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables) 
        actor_grads = tape.gradient(exp_v, self.actor_net.trainable_variables)
        self.optimizer1.apply_gradients(zip(critic_grads, self.critic_net.trainable_variables))
        self.optimizer2.apply_gradients(zip(actor_grads, self.actor_net.trainable_variables)) 