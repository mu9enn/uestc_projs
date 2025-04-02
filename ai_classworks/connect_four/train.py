from stable_baselines3 import PPO
from connect_four_env import ConnectFourGym
from models import CustomCNN

def train_agent(total_timesteps=60000):
    env = ConnectFourGym(agent2="random")
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    print('Device:', model.device)

    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_connectfour")
    return model

if __name__ == '__main__':
    trained_model = train_agent()
