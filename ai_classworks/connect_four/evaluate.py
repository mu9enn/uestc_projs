import numpy as np
from kaggle_environments import make, evaluate
from stable_baselines3 import PPO
from agent import agent1


def get_win_percentages(agent1_func, agent2, n_rounds=100):
    # Default Connect Four configuration
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    outcomes = evaluate("connectx", [agent1_func, agent2], config, [], n_rounds // 2)
    outcomes += [[b, a] for [a, b] in evaluate("connectx", [agent2, agent1_func], config, [], n_rounds - n_rounds // 2)]

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
