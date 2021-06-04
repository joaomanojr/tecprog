# This code is heavily based on Phil Tabor Q learning:
# https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/q_learning/q_learning_agent.py

# Here it were added a DeepQ network also based on Phil Tabor's code:
# https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DQN/deep_q_network.py

# Some hints on how to train the Neural Networks were obtained in this example from begooboi Reddit
# user (based on Keras):
# https://www.reddit.com/r/learnmachinelearning/comments/9vasqm/review_my_code_dqn_for_gym_frozenlake/


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

        # Phil Tabor: self.parameters() from inherited class Module
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # Phil Tabor: pytorch have different tensors for cuda/cpu devices
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        # Phil Tabor: MSELoss will take care of activation for us...
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

        # print_snapshot constants
        self.action_str = ['<', '.', '>', '^']
        self.map_str = []
        self.map_str.append(['S', 'F', 'F', 'F'])
        self.map_str.append(['F', 'H', 'F', 'H'])
        self.map_str.append(['F', 'F', 'F', 'H'])
        self.map_str.append(['H', 'F', 'F', 'G'])

    def one_hot_state(self, state):
        state_m = np.zeros((1, self.input_dims))
        state_m[0][state] = 1
        return state_m

    def choose_action(self, observation):
        ''' Choose Epsilon Greedy action for a given state '''
        rand_ = np.random.random()

        if rand_ > self.epsilon:
            state = T.tensor(self.np_arrays[observation], dtype=T.float).to(self.Q.device)
            # https://stackoverflow.com/questions/64192810/runtime-error-both-arguments-to-matmul-need-to-be-at-least-1d-but-they-are-0d
            actions = self.Q.forward(state.unsqueeze(dim=0))
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

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

        # Phil Tabor: Backpropagate cost and add a step on our optimizer.
        #             These two calls are critical for learn loop.
        loss.backward()
        self.Q.optimizer.step()

        self.decrement_epsilon()

    def batch_learn(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.learn(state, action, reward, next_state, done)

    def print_learn_snapshot(self):
        """
        Print a snapshot of learning situation.

        Sample:
        --
        Learn snapshot:
        |S(<) 0.061||F(^) 0.095||F(<)-0.036||F(<) 0.049|
        |F(^) 0.074||H(0) ~~~~ ||F(>)-0.018||H(0) ~~~~ |
        |F(<) 0.000||F(^) 0.043||F(>) 0.037||H(0) ~~~~ |
        |H(0) ~~~~ ||F(<) 0.066||F(<) 0.072||G(1)  \o/ |
        --

        Cell format: <status>(<best_action>)<best_action_value>
          status      - 'S'=start, 'G'=goal, 'F'=frozen, 'H'=hole
          best_action - '<'=start, '.'=down, '>'=right, '^'=up
                      - '1'=reward
          best_value  - Extracted value for best_action from NN tensor
        
        End cells:
          bad ending - ~~~~ (water)
          good ending -  \o/ (happy)

        """

        print('--\nLearn snapshot: ')

        for line in range(4):
            for col in range(4):
                stateT = T.tensor(self.np_arrays[line * 4 + col], dtype=T.float).to(self.Q.device)
                actionsT = self.Q.forward(stateT.unsqueeze(dim=0))
                if self.map_str[line][col] == 'F' or self.map_str[line][col] == 'S':
                    action_max = self.action_str[T.argmax(actionsT).item()]
                    action_max_value = f'{T.max(actionsT).item(): 4.3f}'
                elif self.map_str[line][col] == 'H':
                    action_max = ' '
                    action_max_value = ' ~~~~ '
                else:
                    action_max = '1'
                    action_max_value = '  \o/ '

                print(f'|{self.map_str[line][col]}({action_max}){action_max_value}|', end='')
            print('')
        print('--\n')
