import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

from collections import deque
import random

class LinearDeepQNetwork(nn.Module):
  def __init__(self, lr, n_actions, input):
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
  def __init__(self, lr, n_actions, gamma=0.90,
               epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
    """ Agent init takes:
    --
    lr - alpha learning rate factor
    input_dims - from our environment dimensions
    n_actions - actions space dimension
    gamma - discount factor on MDP rewards
    epsilon - Epsilon Greedy initial value (exploration threshold)
    eps_dec - Epsilon Greedy decrease factor
    eps_min - Epsilon Greedy minimum, final value (must be > 0)
      
    """
    self.lr = lr
    # Joao: hardcoded as 1
    self.input_dims = 16
    self.n_actions = n_actions
    self.gamma = gamma
    self.epsilon = epsilon
    self.eps_dec = eps_dec
    self.eps_min = eps_min
    self.action_space = [i for i in range(self.n_actions)]
    self.np_arrays = []
    for i in range(self.input_dims):
      self.np_arrays.append(self.one_hot_state(i))

    self.memory = deque(maxlen=2000)

    self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

  def one_hot_state(self, state):
    state_m = np.zeros((1, self.input_dims))
    state_m[0][state] = 1
    return state_m

  def choose_action(self, observation):
    ''' Choose Epsilon Greedy action for a given state '''
    rand_ = np.random.random()
    # print(f'rand_ is {rand_}')

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
    ## # Author: backpropagate cost and add a step on our optimizer.
    ## # These two calls are critical for learn loop.
    loss.backward()
## 
    self.Q.optimizer.step()
    self.decrement_epsilon()

  def batch_learn(self, batch_size):
#        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
#            (state, action, reward, next_state, done) = self.memory[i]
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.learn(state, action, reward, next_state, done)

#        self.decrement_epsilon()