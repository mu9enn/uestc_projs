Below is one way to reorganize the code into separate modules. For example, you might structure your project like this:

```
your_project/
├── agent.py
├── connect_four_env.py
├── evaluate.py
├── models.py
├── train.py
└── requirements.txt
```

Below are sample contents for each file.

---

### **1. connect_four_env.py**

This file implements the Connect Four environment as an OpenAI Gym environment.

```python
# connect_four_env.py
import numpy as np
import gym
from kaggle_environments import make, evaluate
from gym import spaces

class ConnectFourGym(gym.Env):
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Define action and observation spaces.
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2, 
                                            shape=(1, self.rows, self.columns),
                                            dtype=int)
        # Define reward range
        self.reward_range = (-10, 1)
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(1, self.rows, self.columns)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # Agent wins
            return 1
        elif done:         # Opponent wins
            return -1
        else:
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Validate move: check if the chosen column is empty at the top.
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:
            self.obs, old_reward, done, info = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:
            reward, done, info = -10, True, {}
        return np.array(self.obs['board']).reshape(1, self.rows, self.columns), reward, done, info
```

---

### **2. models.py**

This file contains the definition of your custom CNN used as a feature extractor for the PPO policy.

```python
# models.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Run a dummy forward pass to determine the size of the output.
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
```

---

### **3. agent.py**

This file implements the agent function that uses a trained model to choose an action. Notice that we pass the model to the agent.

```python
# agent.py
import numpy as np
import random

def agent1(obs, config, model):
    # Reshape the board to match the model's expected input shape (1,6,7)
    board = np.array(obs['board']).reshape(1, config.rows, config.columns)
    col, _ = model.predict(board)
    # Validate the move: check that the chosen column is valid.
    is_valid = (obs['board'][int(col)] == 0)
    if is_valid:
        return int(col)
    else:
        valid_moves = [c for c in range(config.columns) if obs['board'][c] == 0]
        return random.choice(valid_moves)
```

---

### **4. train.py**

This file trains the PPO agent with your custom CNN on the Connect Four environment.

```python
# train.py
from stable_baselines3 import PPO
from connect_four_env import ConnectFourGym
from models import CustomCNN

def train_agent(total_timesteps=60000):
    env = ConnectFourGym(agent2="random")
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_connectfour")
    return model

if __name__ == '__main__':
    trained_model = train_agent()
```

---

### **5. evaluate.py**

This file defines a function to evaluate your agent’s win percentage against a random opponent. It also demonstrates loading the trained model and wrapping the agent function accordingly.

```python
# evaluate.py
import numpy as np
from kaggle_environments import make, evaluate
from stable_baselines3 import PPO
from agent import agent1

def get_win_percentages(agent1_func, agent2, n_rounds=100):
    # Default Connect Four configuration
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    outcomes = evaluate("connectx", [agent1_func, agent2], config, [], n_rounds//2)
    outcomes += [[b, a] for [a, b] in evaluate("connectx", [agent2, agent1_func], config, [], n_rounds - n_rounds//2)]
    
    win_percentage_agent1 = np.round(outcomes.count([1, -1]) / len(outcomes), 2)
    win_percentage_agent2 = np.round(outcomes.count([-1, 1]) / len(outcomes), 2)
    invalid_agent1 = outcomes.count([None, 0])
    invalid_agent2 = outcomes.count([0, None])
    
    print("Agent 1 Win Percentage:", win_percentage_agent1)
    print("Agent 2 Win Percentage:", win_percentage_agent2)
    print("Number of Invalid Plays by Agent 1:", invalid_agent1)
    print("Number of Invalid Plays by Agent 2:", invalid_agent2)

if __name__ == '__main__':
    # Load the trained model
    model = PPO.load("ppo_connectfour")
    
    # Wrap the agent function to include the model parameter.
    def agent1_wrapper(obs, config):
        return agent1(obs, config, model)
    
    get_win_percentages(agent1_func=agent1_wrapper, agent2="random", n_rounds=100)
```

---

### **6. requirements.txt**

Make sure to list your project dependencies (you can adjust versions as needed):

```
numpy
pandas
matplotlib
gym==0.21.0
kaggle-environments
torch
stable-baselines3
```

---

### **Usage Summary**

1. **Training the Agent:**  
   Run `train.py` to train and save your PPO model.
   ```bash
   python train.py
   ```

2. **Evaluating the Agent:**  
   After training, run `evaluate.py` to test your agent’s performance against a random opponent.
   ```bash
   python evaluate.py
   ```

By splitting your code into separate modules, you maintain a clear, modular design that is easier to test, debug, and maintain over time.Below is one way to reorganize the code into separate modules. For example, you might structure your project like this:

```
your_project/
├── agent.py
├── connect_four_env.py
├── evaluate.py
├── models.py
├── train.py
└── requirements.txt
```

Below are sample contents for each file.

---

### **1. connect_four_env.py**

This file implements the Connect Four environment as an OpenAI Gym environment.

```python
# connect_four_env.py
import numpy as np
import gym
from kaggle_environments import make, evaluate
from gym import spaces

class ConnectFourGym(gym.Env):
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Define action and observation spaces.
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2, 
                                            shape=(1, self.rows, self.columns),
                                            dtype=int)
        # Define reward range
        self.reward_range = (-10, 1)
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(1, self.rows, self.columns)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # Agent wins
            return 1
        elif done:         # Opponent wins
            return -1
        else:
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Validate move: check if the chosen column is empty at the top.
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:
            self.obs, old_reward, done, info = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:
            reward, done, info = -10, True, {}
        return np.array(self.obs['board']).reshape(1, self.rows, self.columns), reward, done, info
```

---

### **2. models.py**

This file contains the definition of your custom CNN used as a feature extractor for the PPO policy.

```python
# models.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Run a dummy forward pass to determine the size of the output.
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
```

---

### **3. agent.py**

This file implements the agent function that uses a trained model to choose an action. Notice that we pass the model to the agent.

```python
# agent.py
import numpy as np
import random

def agent1(obs, config, model):
    # Reshape the board to match the model's expected input shape (1,6,7)
    board = np.array(obs['board']).reshape(1, config.rows, config.columns)
    col, _ = model.predict(board)
    # Validate the move: check that the chosen column is valid.
    is_valid = (obs['board'][int(col)] == 0)
    if is_valid:
        return int(col)
    else:
        valid_moves = [c for c in range(config.columns) if obs['board'][c] == 0]
        return random.choice(valid_moves)
```

---

### **4. train.py**

This file trains the PPO agent with your custom CNN on the Connect Four environment.

```python
# train.py
from stable_baselines3 import PPO
from connect_four_env import ConnectFourGym
from models import CustomCNN

def train_agent(total_timesteps=60000):
    env = ConnectFourGym(agent2="random")
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_connectfour")
    return model

if __name__ == '__main__':
    trained_model = train_agent()
```

---

### **5. evaluate.py**

This file defines a function to evaluate your agent’s win percentage against a random opponent. It also demonstrates loading the trained model and wrapping the agent function accordingly.

```python
# evaluate.py
import numpy as np
from kaggle_environments import make, evaluate
from stable_baselines3 import PPO
from agent import agent1

def get_win_percentages(agent1_func, agent2, n_rounds=100):
    # Default Connect Four configuration
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    outcomes = evaluate("connectx", [agent1_func, agent2], config, [], n_rounds//2)
    outcomes += [[b, a] for [a, b] in evaluate("connectx", [agent2, agent1_func], config, [], n_rounds - n_rounds//2)]
    
    win_percentage_agent1 = np.round(outcomes.count([1, -1]) / len(outcomes), 2)
    win_percentage_agent2 = np.round(outcomes.count([-1, 1]) / len(outcomes), 2)
    invalid_agent1 = outcomes.count([None, 0])
    invalid_agent2 = outcomes.count([0, None])
    
    print("Agent 1 Win Percentage:", win_percentage_agent1)
    print("Agent 2 Win Percentage:", win_percentage_agent2)
    print("Number of Invalid Plays by Agent 1:", invalid_agent1)
    print("Number of Invalid Plays by Agent 2:", invalid_agent2)

if __name__ == '__main__':
    # Load the trained model
    model = PPO.load("ppo_connectfour")
    
    # Wrap the agent function to include the model parameter.
    def agent1_wrapper(obs, config):
        return agent1(obs, config, model)
    
    get_win_percentages(agent1_func=agent1_wrapper, agent2="random", n_rounds=100)
```

---

### **6. requirements.txt**

Make sure to list your project dependencies (you can adjust versions as needed):

```
numpy
pandas
matplotlib
gym==0.21.0
kaggle-environments
torch
stable-baselines3
```

---

### **Usage Summary**

1. **Training the Agent:**  
   Run `train.py` to train and save your PPO model.
   ```bash
   python train.py
   ```

2. **Evaluating the Agent:**  
   After training, run `evaluate.py` to test your agent’s performance against a random opponent.
   ```bash
   python evaluate.py
   ```

By splitting your code into separate modules, you maintain a clear, modular design that is easier to test, debug, and maintain over time.