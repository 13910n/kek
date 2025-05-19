import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from game2 import GoUrEnv
import time
import matplotlib.pyplot as plt
import pickle

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA доступен")
else:
    device = torch.device("cpu")
    print("CUDA недоступен, используется CPU")


# Function to smooth the metrics
def smooth(y, window_size):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')


def plot_metrics(episode_rewards, times, episode_lengths, win_rates, losses, q_values):
    episodes = range(len(episode_rewards))

    plt.figure(figsize=(15, 10))

    plt.subplot(4, 2, 1)
    plt.plot(episodes, episode_rewards, label="Total Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards per Episode")
    plt.legend()

    plt.subplot(4, 2, 3)
    plt.plot(episodes, times, label="Elapsed Time per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Time (seconds)")
    plt.title("Elapsed Time per Episode")
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(episodes, episode_lengths, label="Episode Length")
    plt.xlabel("Episodes")
    plt.ylabel("Length")
    plt.title("Episode Lengths")
    plt.legend()

    plt.subplot(4, 2, 5)
    plt.plot(range(len(win_rates)), win_rates, label="Win Rate")
    plt.xlabel("100 Episode Intervals")
    plt.ylabel("Win Rate")
    plt.title("Win Rate")
    plt.legend()

    smoothed_losses = smooth(losses, 10000)
    plt.subplot(4, 2, 6)
    plt.plot(range(len(smoothed_losses)), smoothed_losses, label="Loss")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()

    smoothed_q_values = smooth(q_values, 10000)
    plt.subplot(4, 2, 7)
    plt.plot(range(len(smoothed_q_values)),
             smoothed_q_values, label="Average Q Value")
    plt.xlabel("Updates")
    plt.ylabel("Average Q Value")
    plt.title("Average Q Value Over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_model(agent, filename):
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'target_model_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
    }, filename)


def load_model(agent, filename):
    checkpoint = torch.load(filename)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Обучающие параметры
EPISODES = 50000
GAMMA = 0.8
ALPHA = 1e-4
EPSILON_START = 0.9
EPSILON_END = 0.1
EPSILON_DECAY = 0.999
BATCH_SIZE = 24
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
TAU = 0.005


class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)
        self.update_target_model()
        self.losses = []
        self.q_values = []

    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                TAU * param.data + (1 - TAU) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        torch_state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(torch_state).numpy()

        valid_act_values = [act_values[0][action['piece_id'][1]]
                            for action in valid_actions]
        return valid_actions[np.argmax(valid_act_values)]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = self.model(state)[0][action]
            if done:
                target = reward
            else:
                next_action = self.model(next_state).max(1)[1].view(1, 1)
                target = reward + GAMMA * \
                    self.target_model(next_state).gather(1, next_action).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
            self.q_values.append(target)
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def get_metrics(self):
        return self.losses, self.q_values


def train():
    env = GoUrEnv()
    agent1 = DoubleDQNAgent(env.state_size, env.action_space_n)
    agent2 = DoubleDQNAgent(env.state_size, env.action_space_n)

    # load_model(agent1, './agent1_model_final.pth')
    # load_model(agent2, './agent2_model_final.pth')

    agent1_wins = 0
    a1_wr = []
    agent2_wins = 0
    a2_wr = []
    episode_lengths = []
    a1_tr = []
    a2_tr = []
    elapsed_times = []

    for e in range(EPISODES):
        start_time = time.time()
        env.reset()
        turn = True
        replay1 = 0
        replay2 = 0
        a1_r = 0
        a2_r = 0
        for score in range(500):
            current_agent = agent1 if turn else agent2
            roll = env.roll()
            valid_actions = env.get_possible_actions(0 if turn else 1, roll)
            if len(valid_actions[0]) == 0:
                continue
            state = env.parse_state(turn, roll)
            action = current_agent.act(state, valid_actions[0])
            next_state, reward, done, is_double = env.step(action)
            if turn:
                a1_r += reward
            else:
                a2_r += reward
            if action is not None:
                current_agent.remember(
                    state, action['piece_id'][1], reward, next_state, done)
                if turn:
                    replay1 += 1
                else:
                    replay2 += 1
            if done:
                a1_tr.append(a1_r)
                a2_tr.append(a2_r)
                if turn:
                    agent1_wins += 1
                else:
                    agent2_wins += 1
                if e % 100 == 0:
                    agent1.update_target_model()
                    agent2.update_target_model()
                    if turn:
                        a1_wr.append(agent1_wins / (e + 1))
                    else:
                        a2_wr.append(agent2_wins / (e + 1))
                episode_lengths.append(score)
                print(
                    f"Episode: {e}/{EPISODES}, Score: {score}, Epsilon1: {agent1.epsilon:.2}, Epsilon2: {agent2.epsilon:.2}")
                break
            if replay1 > BATCH_SIZE:
                replay1 = 0
                agent1.replay(BATCH_SIZE)
            if replay2 > BATCH_SIZE:
                replay2 = 0
                agent2.replay(BATCH_SIZE)
            if not is_double:
                turn = not turn
        end_time = time.time()
        elapsed_times.append(end_time - start_time)

        if e % 1000 == 0:
            save_model(agent1, f'agent1_model_{e}.pth')
            save_model(agent2, f'agent2_model_{e}.pth')

    save_model(agent1, f'agent1_model_final.pth')
    save_model(agent2, f'agent2_model_final.pth')

    a1_losses, a1_qvalues = agent1.get_metrics()

    print("Обучение завершено!")

    with open("a1_qvalues.pkl", "wb") as fp:
        pickle.dump(a1_qvalues, fp)
    with open("a1_tr.pkl", "wb") as fp:
        pickle.dump(a1_tr, fp)
    with open("elapsed_times.pkl", "wb") as fp:
        pickle.dump(elapsed_times, fp)
    with open("episode_lengths.pkl", "wb") as fp:
        pickle.dump(episode_lengths, fp)
    with open("a1_losses.pkl", "wb") as fp:
        pickle.dump(a1_losses, fp)
    with open("a1_wr.pkl", "wb") as fp:
        pickle.dump(a1_wr, fp)

    plot_metrics(a1_tr, elapsed_times, episode_lengths,
                 a1_wr, a1_losses, a1_qvalues)


if __name__ == "__main__":
    train()
