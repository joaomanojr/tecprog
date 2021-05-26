import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

from collections import deque
import random

class LinearDeepQNetwork(nn.Module):
  def __init__(self, lr, input, n_actions):
    super(LinearDeepQNetwork, self).__init__()

    self.fc1 = nn.Linear(input, 128)
    self.fc2 = nn.Linear(128, n_actions)

    # Author: self.parameters() from inherited class Module
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    # Author: pytorch have different tensors for cuda/cpu devices
    self.to(self.device)

  def forward(self, state):
    layer1 = F.relu(self.fc1(state))
    # Author: MSELoss will take care of activation for us...
    actions = self.fc2(layer1)

    return actions


class Agent():
  def __init__(self, lr, state_space, action_space, gamma=0.90,
               epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
    """ Agent init takes:
    --
    lr - alpha learning rate factor
    state_space - environment state space dimension
    action_space - environment actions space dimension
    gamma - discount factor on MDP rewards
    epsilon - Epsilon Greedy initial value (exploration threshold)
    eps_dec - Epsilon Greedy decrease factor
    eps_min - Epsilon Greedy minimum, final value (must be > 0)
      
    """
    self.lr = lr
    self.input_dims = state_space
    self.n_actions = action_space
    self.gamma = gamma
    self.epsilon = epsilon
    self.eps_dec = eps_dec
    self.eps_min = eps_min
    self.action_space = [i for i in range(self.n_actions)]
    self.np_arrays = []
    for i in range(self.input_dims):
      self.np_arrays.append(self.one_hot_state(i))

    self.memory = deque(maxlen=2000)

    self.Q = LinearDeepQNetwork(self.lr, self.input_dims, self.n_actions)

  def one_hot_state(self, state):
    state_m = np.zeros((1, self.input_dims))
    state_m[0][state] = 1
    return state_m

  def choose_action(self, observation):
    ''' Choose Epsilon Greedy action for a given state '''
    rand_ = np.random.random()

    if rand_ > self.epsilon:
      state = T.tensor(self.np_arrays[observation], dtype=T.float).to(self.Q.device)
      ## print(f'state is {state}')
      ## print(f'state is {state.unsqueeze(dim=0)}')
      # https://stackoverflow.com/questions/64192810/runtime-error-both-arguments-to-matmul-need-to-be-at-least-1d-but-they-are-0d
      actions = self.Q.forward(state.unsqueeze(dim=0))
      action = T.argmax(actions).item()
    else:
      action = np.random.choice(self.action_space)
      ## print(f'random action {action}')

    return action
    
  def decrement_epsilon(self):
    ''' Epsilon decrease function (linear) '''
    # Look: my beloved C ternary in python terms!
    self.epsilon = self.epsilon - self.eps_dec \
                      if self.epsilon > self.eps_min else self.eps_min

  def learn(self, state, action, reward, state_, done):
    """ Off Policy (always Greedy) Learn function 
    --
    Here defined as plain Bellman equation, state_ is state'
    """
    self.Q.optimizer.zero_grad()
    state_T = T.tensor(self.np_arrays[state_], dtype=T.float).to(self.Q.device)

    #print(f'reward is {reward}')
    if not done:
      actions_T = self.Q.forward(state_T.unsqueeze(dim=0))
      rewardT = reward + self.gamma * T.max(actions_T)
    else:
      rewardT = T.tensor(reward, dtype=T.float).to(self.Q.device)

    stateT = T.tensor(self.np_arrays[state], dtype=T.float).to(self.Q.device)

    q_pred = self.Q.forward(stateT.unsqueeze(dim=0))
    q_target = self.Q.forward(stateT.unsqueeze(dim=0))
    q_target[0][0][action] = rewardT
    loss = self.Q.loss(q_pred, q_target).to(self.Q.device)

    # Author: backpropagate cost and add a step on our optimizer.
    # These two calls are critical for learn loop.
    loss.backward()
    self.Q.optimizer.step()

    self.decrement_epsilon()

  def batch_learn(self, batch_size):
      minibatch = random.sample(self.memory, batch_size)
      for state, action, reward, next_state, done in minibatch:
        self.learn(state, action, reward, next_state, done)

  def print_learn_snapshot(self):
    """
    Print a snapshot of learning situation

    |S(.)0.589||F(>)0.653||F(.)0.727||F(<)0.668|
    |F(.)0.655||H(.)0.677||F(.)0.809||H(.)0.637|
    |F(>)0.728||F(>)0.810||F(.)0.900||H(>)0.808|
    |H(.)0.684||F(>)0.898||F(>)0.997||G(.)0.657|

    Cell format: <status>(<best_action>)<best_action_value>
      status      - 'S'=start, 'G'=goal, 'F'=frozen, 'H'=hole
      best_action - '<'=start, '.'=down, '>'=right, '^'=up
      best_value  - Extracted value for best_action from NN tensor
    """

    print('\nLearn snapshot: ')

    # actions:: LEFT = 0 DOWN = 1 RIGHT = 2 UP = 3
    action_str = ['<', '.', '>', '^']

    map_str = []
    map_str.append(['S', 'F', 'F', 'F'])
    map_str.append(['F', 'H', 'F', 'H'])
    map_str.append(['F', 'F', 'F', 'H'])
    map_str.append(['H', 'F', 'F', 'G'])

    for line in range(4):
      for col in range(4):
        stateT = T.tensor(self.np_arrays[line * 4 + col], dtype=T.float).to(self.Q.device)
        actionsT = self.Q.forward(stateT.unsqueeze(dim=0))
        print(f'|{map_str[line][col]}({action_str[T.argmax(actionsT).item()]}){T.max(actionsT).item():4.3f}|', end='')
      print('')
