import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from model import DuelingDQN


import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
PRIO_E = 1e-5           # e parameter for prioritized Experience Replay
PRIO_A = 0.6            # a parameter for prioritized Experience Replay
PRIO_B = 0.4            # b parameter for prioritized Experience Replay
PRIO_B_INC = 0.000005     # b parameter increment

# Hassan : ToDo
# dropput removed : does not work well
# Currently the prioritized queue works after certain iteration. dependent upon b_step.
# Looks ok as in the beginning until the buffer is full its better to take equal probability

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,max_t=1000):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingDQN(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingDQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.prio_b = PRIO_B
        self.b_step = 0
        self.max_b_step = 2000
        self.learnFirst = True

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)

        # Hassan : Save the experience in prioritized replay memory
        self.memory.prio_add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                #experiences = self.memory.sample()
                #self.learn(experiences, GAMMA)

                # Hassan : prioritized replay memory

                self.b_step = self.b_step + 1
                experiences, indices = self.memory.prio_sample()
                self.learn(experiences, GAMMA, indices)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def get_beta(self, t):
        '''
        Return the current exponent β based on its schedul. Linearly anneal β
        from its initial value β0 to 1, at the end of learning.
        :param t: integer. Current time step in the episode
        :return current_beta: float. Current exponent beta
        '''
        #f_frac = min(float(t) / self.max_b_step, 1.0)
        #current_beta = self.prio_b + f_frac * (1. - self.prio_b)
        #current_beta = min(1,current_beta)
        self.prio_b = min(1,self.prio_b + PRIO_B_INC)
        return self.prio_b

    def learn(self, experiences, gamma, indices):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, probabilities = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"

        # Get max predicted Q values (for next states) from target model
        # Hassan : Action is selected using greedy policy
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Hassan : Double DQN
        # Selecting actions which maximizes while taking w (qnetwork_local)
        next_actions = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
        #next_actions_test = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1) # Hassan : from the example
        #print(torch.sum(next_actions-next_actions_test)) # Hassan : no difference found
        # Selecting q values of these actions using w' (qnetwork_target)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)


        # Compute Q targets for current states
        # Hassan : This is TD Target
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # Hassan : This is current value
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        #Hassan : Compute the td_error
        td_error = Q_targets - Q_expected
        #print(td_error.detach().numpy())
        #self.prio_b = min(1, PRIO_B_INC+self.prio_b)
        f_currbeta = self.get_beta(0)
        #print(f_currbeta)
        #f_currbeta = self.get_beta(self.b_step)
        #print(self.b_step)

        #print(t)
        #print(self.prio_b)
        weights_importance = probabilities.mul_(self.memory.__len__()).pow_(-f_currbeta)
        #  Hassan : calculate max_weights_importance
        #probabilities_min = min(self.memory.priorities)/self.memory.cum_priorities
        probabilities_min = self.memory.min_priority/self.memory.cum_priorities
        max_weights_importance = (probabilities_min * self.memory.__len__())**(-f_currbeta)
        # Hassan : divide the weights importance with the max_weights_importance
        # Hassan : Improvement why not calculating the max_weights_importance = max(weights_importance)??
        # Hassan : this will only calculating on the current list not the complete one

        #print(weights_importance)
        #print(weights_importance.max(0)[0])
        #print(max_weights_importance)
        #if self.learnFirst:
        #    self.learnFirst = False
        #else :
        #    max_weights_importance = max_weights_importance[0]

        weights_final = weights_importance.div_(max_weights_importance)

        square_weighted_error = td_error.pow_(2).mul_(weights_final)
        loss = square_weighted_error.mean()

        # Hassan : after the observations observation from example, update was done after the weights calculation
        #if self.prio_b > 0.5:
        self.memory.prio_update(indices,td_error.detach().numpy(),PRIO_E,PRIO_A)


        # Compute loss
        #loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # Hassan : Here not after C steps w is changed though cahnged slightly after every learn step
        # Hassan : We can modify to change this after ever C steps
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        #self.prio_experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.cum_priorities = 0.
        self._buffer_size = buffer_size
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.
        self.min_priority = 1.


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    def prio_add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        # exclude the value that will be discareded

        if len(self.priorities) >= self._buffer_size:
            self.cum_priorities -= self.priorities[0]
            #if self.priorities[0] < self.min_priority:
            #    self.min_priority = min(self.priorities)

        # include the max priority possible initialy
        self.priorities.append(self.max_priority)  # already use alpha
        # accumulate the priorities abs(td_error)
        self.cum_priorities += self.priorities[-1]




    def prio_sample(self):
        """Randomly sample a batch of experiences from memory."""
        probabilities_all = None
        if self.cum_priorities:
            probabilities_all = np.array(self.priorities)/self.cum_priorities

        #indices = random.choices(np.arange(self.__len__()), weights=probabilities_all, k=self.batch_size)
        #print(self.priorities)
        #print(probabilities_all)

        buffer_len = self.__len__()
        indices = np.random.choice(buffer_len,size=min(buffer_len, self.batch_size),p=probabilities_all)

        states = torch.from_numpy(np.vstack([self.memory[idx].state for idx in indices if idx is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[idx].action for idx in indices if idx is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx].reward for idx in indices if idx is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[idx].next_state for idx in indices if idx is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[idx].done for idx in indices if idx is not None]).astype(np.uint8)).float().to(device)
        probabilities = torch.from_numpy(np.vstack([probabilities_all[idx] for idx in indices if idx is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones, probabilities) , indices

    def prio_update(self, indices, td_error, e=1e-3, a=0.7):
        for i, f_tderr in zip(indices, td_error):
            self.cum_priorities -= self.priorities[i]
            # transition priority: pi^α = (|δi| + ε)^α
            self.priorities[i] = np.power(np.abs(f_tderr[0]) + e,a)
            self.cum_priorities += self.priorities[i]

            # Hassan : check whether this has some problem?
            if self.priorities[i]>self.max_priority:
                self.max_priority = self.priorities[i]
            if self.priorities[i]<self.min_priority:
                self.min_priority = self.priorities[i]
        #self.max_priority = max(self.priorities)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
