# This code is heavily based on Phil Tabor Q learning:
# https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/q_learning/frozen_lake_q_learning.py
#
# On this level main difference is to learn in batches as implmented by begooboi Reddit user:
# https://www.reddit.com/r/learnmachinelearning/comments/9vasqm/review_my_code_dqn_for_gym_frozenlake/


import gym
import matplotlib.pyplot as plt
import numpy as np
from deepq_learning_agent import Agent

if __name__ == '__main__':

    # TODO(joao): remove is_slippery = False when NN is improved
    env = gym.make('FrozenLake-v0', is_slippery=False)

    n_games = 2000
    scores = []
    win_pct_list = []
    batch_size = 100

    agent = Agent(lr=0.0005, state_space=env.observation_space.n, action_space=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        # Initialize environment
        state = env.reset()

        # Interact with environment until done (end or fall in a hole)
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            # env.render()
            score += reward
            agent.memory.append((state, action, reward, state_, done))
            state = state_
        scores.append(score)

        if len(agent.memory) > batch_size:
            agent.batch_learn(batch_size)

        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            print(f'episode {i} win pct {win_pct:.2f} epsilon {agent.epsilon:.2f}')
            agent.print_learn_snapshot()

    plt.plot(win_pct_list)
    plt.show()
