import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np

# 定义超参数
learning_rate = 0.002
gamma = 0.99
lmbda = 0.95
eps_clip = 0.2
K_epoch = 3
T_horizon = 20

# 定义PPO网络
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1   = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v  = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s_batch = torch.tensor(np.array(s_lst), dtype=torch.float)
        a_batch = torch.tensor(a_lst)
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_batch = torch.tensor(done_lst, dtype=torch.float)
        prob_a_batch = torch.tensor(prob_a_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch, prob_a_batch

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2).mean() + F.smooth_l1_loss(self.v(s), td_target.detach()).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def train():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(3000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                s_tensor = torch.from_numpy(s).float()
                prob = model.pi(s_tensor)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime
                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# Episode: {}, Average Score: {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()
    # 保存模型参数
    torch.save(model.state_dict(), 'ppo_cartpole.pth')
    print("Training completed and model saved.")

def test():
    # 创建带有渲染的环境
    env = gym.make('CartPole-v1', render_mode='human')
    model = PPO()
    # 加载训练好的模型参数
    model.load_state_dict(torch.load('ppo_cartpole.pth'))
    model.eval()

    s, _ = env.reset()
    done = False
    score = 0.0

    while not done:
        s_tensor = torch.from_numpy(s).float()
        prob = model.pi(s_tensor)
        a = prob.argmax().item()
        s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        score += r
        # 环境渲染
        env.render()

    print("Test completed, Score: {:.1f}".format(score))
    env.close()

if __name__ == '__main__':
    train()
    test()