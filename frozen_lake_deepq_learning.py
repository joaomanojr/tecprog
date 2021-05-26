import gym
import matplotlib.pyplot as plt
import numpy as np
from deepq_learning_agent import Agent

if __name__ == '__main__':

  env = gym.make('FrozenLake-v0', is_slippery = False)
  #env = gym.make('FrozenLake-v0')

  n_games = 2000
  scores = []
  win_pct_list = []
  batch_size = 100

  agent = Agent(lr = 0.0005, state_space = env.observation_space.n, action_space = env.action_space.n)

  for i in range(n_games):
    score = 0
    done = False
    # This looks like a context pointer to created environment, here we are
    # just initializing it - it returns our current state also;
    obs = env.reset()

    # Interact with environment until done (end or fall in a hole)
    while not done:
      action = agent.choose_action(obs)
      obs_, reward, done, info = env.step(action)
      #env.render()
      score += reward
      #agent.learn(obs, action, reward, obs_, done)
      agent.memory.append((obs, action, reward, obs_, done))
      obs = obs_
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