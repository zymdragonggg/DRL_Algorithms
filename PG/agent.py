import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp

# np.random.seed(1)
# tf.random.set_seed(3)

#搭建DNN
class MyNN(Model):
    def __init__(self, s_dim, a_dim, hidden_nodes1, hidden_nodes2):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(hidden_nodes1, 'relu') # w: [s_dim, hidden_nodes1], b: [hidden_nodes1, ]
        self.d2 = tf.keras.layers.Dense(hidden_nodes2, 'relu')
        self.d3 = tf.keras.layers.Dense(a_dim, None) 
        self(tf.keras.Input(shape=s_dim))   # 初始化参数 [B, s_dim]

    def call(self, s):
        x = self.d1(s)
        x = self.d2(x)
        x = self.d3(x)
        p = tf.nn.log_softmax(x)
        return p    # [B, A]

#创建agent
class PolicyGradient:
    def __init__(
            self,
            action_dim,
            state_dim,
            learning_rate = 0.01,
            reward_decay = 0.95,
    ):
        self.a_dim = action_dim #动作纬度
        self.s_dim = state_dim #状态纬度
        self.lr = learning_rate #学习率
        self.gamma = reward_decay #打着因子
        self.num_hidden1 = 32 
        self.num_hidden2 = 32

        self.ep_s, self.ep_a, self.ep_r = [], [], []
        self.net = MyNN(self.s_dim, self.a_dim, self.num_hidden1, self.num_hidden2)  
        self.optimizer = tf.keras.optimizers.Adam(self.lr) 
        self.ck_point = tf.train.Checkpoint(policy=self.net) 
        self.saver = tf.train.CheckpointManager(self.ck_point, directory='./models_Cartpole', max_to_keep=5, checkpoint_name='zym_')

    def save_model(self, episode):#保存模型
        self.saver.save(checkpoint_number=episode)

    def restore_model(self, directory):#恢复模型
        self.ck_point.restore(tf.train.latest_checkpoint(directory))
              

    def choose_action(self, s):
        s = s.reshape([1, self.s_dim]) #对传过来的数据升维度（[2,]-->[1,2]）
        action_prob= self.net(s) #将状态输入到网络中获得所有动作的概率
        norm_dist = tfp.distributions.Categorical(probs=tf.math.exp(action_prob)) 
        action = int(norm_dist.sample()) #这一步和上一步用来根据概率选动作，输出为tensor需要变成int
        return action

    def store_transition(self, s, a, r):#保存整个episode
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)

    def learn(self):
        discounted_ep_r = self._discount_and_norm_rewards() #获得在每一步的累计折扣reward
        with tf.GradientTape() as tape:
            s = np.array(self.ep_s) #输入整个序列的所有状态
            all_action_prob = self.net(s) #输出动作概率 
            norm_dist = tfp.distributions.Categorical(probs=tf.math.exp(all_action_prob)) 
            sa_prob = norm_dist.log_prob(self.ep_a) #选出之前在序列中执行的动作的概率
            loss = -tf.reduce_mean(sa_prob * discounted_ep_r) #注意是负号，因为要最大值，但是loss是求最小
        loss_grads = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(loss_grads, self.net.trainable_variables))
        self.ep_s, self.ep_a, self.ep_r = [], [], [] #注意一定要清空。

    def _discount_and_norm_rewards(self): # 
        discounted_ep_rs = np.zeros_like(self.ep_r)
        running_add = 0
        for t in reversed(range(0, len(self.ep_r))):
            running_add = running_add * self.gamma + self.ep_r[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        # print(discounted_ep_rs)
        return discounted_ep_rs   