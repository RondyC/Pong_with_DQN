# Установка зависимостей
"""

!pip install --upgrade pyvirtualdisplay ipykernel > /dev/null 2>&1

!pip install "gymnasium[atari]"
!pip install autorom[accept-rom-license]

"""# Импорт необходимых библиотек"""

import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import copy

"""# Настройка среды"""

gym.register_envs(ale_py)

env = gym.make("PongDeterministic-v4", render_mode='rgb_array')

state_shape = env.observation_space.shape
print('Число состояний (форма):', state_shape)
n_action = env.action_space.n
print('Число действий:', n_action)
print('Доступные действия:', env.unwrapped.get_action_meanings())

image_size = 84

"""# Преобразование состояния"""

def get_state(prev_obs, obs):
    p = torch.from_numpy(prev_obs).float().div(255.0).mean(dim=2)
    c = torch.from_numpy(obs).float().div(255.0).mean(dim=2)
    diff = (c - p).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(diff,
                        size=(image_size, image_size),
                        mode='bicubic',
                        align_corners=False)
    return out.squeeze(0)

"""# Модель DQN (Полносвязанные слои)"""

class DQNModel(nn.Module):
    def __init__(self, n_action):
        super().__init__()
        self.fc1 = nn.Linear(image_size * image_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_action)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

"""# Класс-обёртка для обучения DQN"""

class DQN:
    def __init__(self, n_action, lr=1e-3):
        self.model = DQNModel(n_action)
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def predict(self, state):
        with torch.no_grad():
            return self.model(state.unsqueeze(0)).squeeze(0)

    def replay(self, memory, batch_size, gamma):
        if len(memory) < batch_size:
            return

        batch = random.sample(memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        s_batch  = torch.stack(states)
        ns_batch = torch.stack(next_states)
        a_batch  = torch.tensor(actions)
        r_batch  = torch.tensor(rewards, dtype=torch.float32)
        d_batch  = torch.tensor(dones, dtype=torch.bool)

        q_values = self.model(s_batch)
        q_s_a    = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next      = self.model_target(ns_batch)
            max_q_next  = q_next.max(dim=1)[0]
            target_q    = r_batch + gamma * max_q_next * (~d_batch)

        loss = self.loss_fn(q_s_a, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

"""# ε-жадная стратегия"""

def gen_epsilon_greedy_policy(agent, epsilon, n_action):
    def policy(state):
        if random.random() < epsilon:
            return random.randrange(n_action)
        else:
            q = agent.predict(state)
            return int(torch.argmax(q).item())
    return policy

"""# Параметры обучения"""

n_episodes    = 100
batch_size    = 64
gamma         = 0.95
epsilon       = 1.0
eps_min       = 0.01
eps_decay     = 0.99
target_update = 10

agent = DQN(n_action, lr=1e-4)
memory = deque(maxlen=30000)
rewards_history = []

"""# Основной цикл обучения"""

for ep in range(n_episodes):
    if ep % target_update == 0:
        agent.copy_target()

    prev_obs, _ = env.reset()
    state = get_state(prev_obs, prev_obs)
    total_reward = 0
    done = False

    policy = gen_epsilon_greedy_policy(agent, epsilon, n_action)

    while not done:
        action = policy(state)
        obs, reward, done, truncated, _ = env.step(action)

        next_state = get_state(prev_obs, obs)
        memory.append((state, action, next_state, reward, done or truncated))
        agent.replay(memory, batch_size, gamma)

        prev_obs = obs
        state    = next_state
        total_reward += reward

    rewards_history.append(total_reward)
    epsilon = max(epsilon * eps_decay, eps_min)
    print(f"Эпизод {ep+1}/{n_episodes}, вознаграждение: {total_reward:.1f}, ε={epsilon:.3f}")

"""# График вознаграждений"""

plt.plot(rewards_history)
plt.title("Вознаграждение по эпизодам")
plt.xlabel("Эпизод")
plt.ylabel("Суммарное вознаграждение")
plt.show()

"""# Выводы

---

1. Инструменты и библиотеки:
* Использованы gymnasium (среда PongDeterministic-v4), ale-py для запуска Atari, PyTorch для построения и обучения нейросети, matplotlib для визуализации.
2. Предобработка состояний:
* Кадры преобразовывались в оттенки серого, нормализовались и интерполировались до размера 84×84 пикселя с использованием билинейной интерполяции (F.interpolate).
3. Архитектура модели:
* Агент основан на полносвязной нейронной сети с двумя скрытыми слоями по 512 нейронов каждый. Выходной слой соответствовал числу возможных действий (6).
4. Алгоритм обучения:
* Применён алгоритм Deep Q-Network (DQN) с таргет-сетью (обновление каждые 10 эпизодов), функцией потерь MSE и оптимизатором Adam (lr=1e-4).
5. Стратегия выбора действий:
* Использовалась ε-жадная политика с начальным ε=1.0, минимальным ε=0.01 и затуханием 0.99.
6. Результаты обучения:
* За 100 эпизодов агент достигал суммарного вознаграждения от -21 до -16, что свидетельствует о появлении отдельных успешных эпизодов и локальных улучшений в стратегии поведения.
7. Динамика награды:
* График вознаграждений показывает нестабильное, но местами положительное развитие поведения агента, с признаками способности к обучению на отдельных участках.
"""