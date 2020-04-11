import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.next_action = np.random.choice(self.nA)
        self.gamma=1.0
        self.alpha=0.1 #this can be potentially changed
        self.epsilon=1.0
        self.eps_start=1.0
        self.eps_decay=.99
        self.eps_min=0.0005

    def select_epsilon(self, episode):
        #here epsilon value will be updated for each episode
        self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)
        #self.epsilon = 1.0 / episode


    def get_probs(self,Q_s, epsilon, nA):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s

    def update_Qsa(self,Qsa,Q_next_sa, next_reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent episode """
        old_Q = Qsa
        Qsa = old_Q + alpha*(next_reward + gamma*Q_next_sa - old_Q)
        return Qsa

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        return self.next_action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if done:
            self.Q[state][action] = self.update_Qsa(self.Q[state][action],0,reward,self.alpha, self.gamma)
        else:
            if next_state in self.Q:
                self.next_action = np.random.choice(np.arange(self.nA), p=self.get_probs(self.Q[next_state], self.epsilon, self.nA))
            else :
                self.next_action = np.random.choice(self.nA)
            self.Q[state][action] = self.update_Qsa(self.Q[state][action],self.Q[next_state][self.next_action],reward,self.alpha, self.gamma)
