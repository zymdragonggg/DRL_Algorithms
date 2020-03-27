import gym
from agent import PolicyGradient

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    # env.seed(1)   
    env = env.unwrapped

    # print(env.action_space)
    # print(env.observation_space)
    # print(env.observation_space.high)
    # print(env.observation_space.low) 
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    learning_rate = 0.01
    reward_decay = 0.9

    RL = PolicyGradient(action_dim, state_dim, learning_rate, reward_decay)
    # RL.restore_model('./models_Cartpole') #如果需要验证自己模型，执行这一句
    for episode in range(1000):
        if episode % 10 == 0:#每10步保存一下模型
            RL.save_model(episode)
        s = env.reset()#初始化环境
        print('--------This is the', episode,'episode----------')
        while True:
            a = RL.choose_action(s) 
            s_, r, done, info = env.step(a)
            RL.store_transition(s, a, r)
            if done:
                print('the total reward:', sum(RL.ep_r))
                break
            s = s_
            env.render()
        RL.learn()
           