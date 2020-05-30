import gym
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from gym import error, spaces, utils
from gym.utils import seeding

# Create a custom environment 
class WTAEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, assignment, target_values, prob, device):
        ''' 
        Assignment - 1D array that maps weapons to assigned target
        target_values - 1D array that maps targets to their value when destroyed
        prob - m x n array where prob[i, j] = probability of weapon i killing target j
        ''' 
        super(WTAEnv, self).__init__()
        
        self.n = len(target_values)
        self.m = len(assignment)

        self.start_assignment = assignment
        self.assignment = assignment
        
        # one hot encoded version of assignment, for quick reward computation
        self.assignment_onehot = self.get_onehot_assignments()
        
        self.target_values = target_values 
        self.prob = prob
        # q = probability array of survival
        self.q = 1 - self.prob
        
        # Current expected value of the assignment
        self.value = self.assignment_value()
        
        # The action space - a number 0 <= i < m * n, where
        # (weapon = i // n, target = i % n)
        # Assigns weapon to target (one target per weapon)
        self.action_space = spaces.MultiDiscrete([self.m * self.n])
        
        self.device = device
        
    def decode_action(self, action):
        '''
        Given an action, return the weapon and target associated with
        that action.
        '''
        return action // self.n, action % self.n
        
    def get_onehot_assignments(self):
        '''
        Get a one-hot encoded representation of the assignments.
        This is used only because this representation is convenient and
        fast for computing the assignment value, since we can use
        vectorization and not use for loops. Not sure how much this
        actually helps though.
        '''
        onehot = np.zeros((self.m, self.n))
        onehot[np.arange(onehot.shape[0]), self.assignment] = 1
        return onehot
 
    def assignment_value(self):
        '''
        Compute the expected value of our assignment.
        E = Sum over targets i [P(target i killed) * Value(i)]
        where P(target i killed) = 1 - P(i survives)
        where P(i surves) = 1 - Product over weapons j [P(i survives j) = q[i, j]]
        '''
        pkill = 1 - np.prod(self.q ** self.assignment_onehot, axis=0)
        expected_value = np.dot(pkill, self.target_values)
        return expected_value
        
    def step(self, action):
        '''Perform action on the current state'''
        weapon, target = self.decode_action(action[0])
        if weapon < 0 or weapon >= self.m or target < 0 or target >= self.n:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        # Update assignments
        old_target = self.assignment[weapon]
        self.assignment[weapon] = target
        self.assignment_onehot[weapon, old_target] = 0
        self.assignment_onehot[weapon, target] = 1

        # Reward is change in assignment value
        new_value = self.assignment_value()
        reward = new_value - self.value
        self.value = new_value

        # There is no stopping condition other than the max number of iterations
        done = False

        return (self.get_state(), reward, done, {})

    def get_state(self):
        '''
        Our state is a size m + n array.
        state[:m] is self.assignments
        state[m:] is self.target_values
        '''
        state = np.concatenate([self.assignment, self.target_values])
        return torch.tensor(state, device=self.device)

    def reset(self):
        '''
        Important: the observation must be a numpy array
        :return: (np.array) 
        '''
        self.assignment = self.start_assignment
        self.assignment_onehot = self.get_onehot_assignments()
        return self.get_state()

    def render(self, mode='human', close=False):
        pass

class SumTree(object):
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.fill = 0
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=Transition )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx 

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        assert isinstance(data, Transition)
        self.data[self.write] = data
        self.update(idx, p)
        self.fill += 1
        self.write += 1
        if self.fill >= self.capacity:
            self.fill = self.capacity - 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
    
    def len(self):
        return self.fill

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = 0.6
        self.min_priority = 0.01

    def push(self, priority, transition):
        """Saves a transition."""
        assert isinstance(transition, Transition)
        self.tree.add(np.power(priority + self.min_priority, self.alpha), transition)

    def sample(self, batch_size):
        sampled = set()
        transitions = []
        ids = []
        num_sample = 0
        while num_sample < batch_size:
            s = np.random.uniform(0, self.tree.total())
            data = self.tree.get(s)
            if data[0] not in sampled:
                sampled.add(data[0])
                assert isinstance(data[2], Transition)
                assert isinstance(data[0], int)
                transitions.append(data[2])
                ids.append(data[0])
                num_sample += 1
        return transitions, ids

    def update(self, idx, priority):
        self.tree.update(idx, priority)

    def num_stored(self):
      return self.tree.len()

'''Take in n by m matrix, convert it to 1D feature vector '''
class DQN(nn.Module):
    def __init__(self, n, m, embedding_size=8):
        super(DQN, self).__init__()
        # The assignment becomes embedded, so it has size m * embedding_size
        # when flattened
        # The n comes from the values attached
        self.assignment_size = m * embedding_size
        self.input_size = self.assignment_size + n
        self.output_size = m * n
        self.n = n
        self.m = m
        
        units = 128

        self.embedding_size = embedding_size
        # Embed the targets, since the actual numerical value of the
        # targets don't mean anything
        # Another idea: skip the middleman and replace the targets
        # with the target values
        self.embedding = nn.Embedding(n, self.embedding_size)
        self.lin1 = nn.Linear(self.input_size, units)
        self.drop1 = nn.Dropout(0.3)
        self.lin2 = nn.Linear(units, self.output_size)
        self.drop2 = nn.Dropout(0.2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        assignment = state[:, :self.m].long()
        assignment = self.embedding(assignment)
        
        values = state[:, self.m:].float()
                
        # Flatten the assignment embedding
        assignment = assignment.view(-1, self.assignment_size).float() 
        
        # and concatenate the values
        x = torch.cat([assignment, values], dim=1)
        
        x = F.relu(self.drop1(self.lin1(x)))
        x = F.relu(self.lin2(x))
        return x

# with dueling networks
class DuelingDQN(nn.Module):

    def __init__(self, n, m, embedding_size=8, units=128):
        super(DuelingDQN, self).__init__()
        # The assignment becomes embedded, so it has size m * embedding_size
        # when flattened
        # The n comes from the values attached
        self.assignment_size = m * embedding_size
        self.input_size = self.assignment_size + n
        self.output_size = m * n
        self.n = n
        self.m = m
      
        self.units = units

        self.embedding_size = embedding_size
        # Embed the targets, since the actual numerical value of the
        # targets don't mean anything
        # Another idea: skip the middleman and replace the targets
        # with the target values
        self.embedding = nn.Embedding(n, self.embedding_size)

        # Layer to measure the value of a state
        self.value_stream = nn.Sequential(
            nn.Linear(self.input_size, units),
            nn.ReLU(),
            nn.Linear(units, 1)
        )
        # Layer to measure the advantages of an action given a state
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.input_size, units),
            nn.ReLU(),
            nn.Linear(units, self.output_size)
        )

    def forward(self, state):
        assignment = state[:, :self.m].long()
        assignment = self.embedding(assignment)


        values = state[:, self.m:].float()

        # Flatten the assignment embedding
        assignment = assignment.view(-1, self.assignment_size).float() 
        
        # and concatenate the values
        x = torch.cat([assignment, values], dim=1)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = values + (advantages - advantages.mean())
        
        return qvals

    def feature_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)

def generate_initial_assignment(n, m):
    '''   
    Randomly assign weapons to targets
    ''' 
    assignment = np.random.randint(n, size=m)
    return assignment

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


steps_done = 0

def is_possible(state, weapon, target):
    '''
    We don't want to assign a weapon to the target if it is already assigned
    to the target, since this does not change the state at all.
    '''
    curr_target = state[weapon].item()
    return curr_target != target 

'''
Takes a state and returns an action and boolean which is True iff the model exploited
'''
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        ## Exploit
        with torch.no_grad():
            policy_net.eval()
            state_batch = torch.unsqueeze(state, 1).transpose(0, 1).float()
            largest = torch.sort(policy_net(state_batch), descending=True, dim=1)[1]
            policy_net.train()
            
            # Try until we get a valid action
            for i in largest[0]:
                weapon = i / n
                target = i % n
                
                if is_possible(state, weapon.item(), target.item()):
                    return torch.tensor([i], device=device), True
                
            # This should never happen
            raise ValueError('Invalid state: no possible action')
    else:
        ## Explore
        weapon = np.random.randint(m)
        curr_target = state[weapon].item()
        target = np.random.randint(n)
        while target == curr_target:
            target = np.random.randint(n)
        
        action = weapon * n + target
        return torch.tensor([action], device=device, dtype=torch.long), False

'''
Takes a list of transitions and returns a list of new priority values
'''
def get_error(transitions, policy_net, target_net):
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
        
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(len(transitions), device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#     print(next_state_values[non_final_mask].shape)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss

    loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())
    if len(transitions) == 1:
        return loss
    else:
        priority = [F.smooth_l1_loss(state_action_values[i].float(), expected_state_action_values[i].float()) for i in range(len(transitions))]
        return priority, loss

def optimize_model():
    if memory.num_stored() < BATCH_SIZE:
        return
    transitions, ids = memory.sample(BATCH_SIZE) 
    priority, loss = get_error(transitions, policy_net, target_net)
    for i in range(len(ids)):
      memory.update(ids[i], priority[i])
    
    value_loss = loss.item()
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return value_loss

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50
TARGET_UPDATE = 200
MAX_ITERATIONS = 500

# n - number of targets
n = 4
# m - number of weapons
m = 5
assert n > 1

lower_val = 25
upper_val = 50
lower_prob = 0.6
upper_prob = 0.9
values = np.random.uniform(lower_val, upper_val, n)
prob = np.random.uniform(lower_prob, upper_prob, (m, n))
assignment = generate_initial_assignment(n, m)
env = WTAEnv(assignment, values, prob, device)

#policy_net = DQN(n, m, n // 2).to(device)
#target_net = DQN(n, m, n // 2).to(device)
policy_net = DuelingDQN(n, m).to(device)
target_net = DuelingDQN(n, m).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

num_episodes = 16
env.reset()
init_state = env.get_state() 
for i_episode in range(1, num_episodes+1):
    # Initialize the environment and state
    env.reset()
    state = init_state
    for t in range(1, MAX_ITERATIONS+1):
        print(f'episode {i_episode}/{num_episodes}, iteration {t}/{MAX_ITERATIONS}', end=' ')
        # Select and perform an action
        action, exploit = select_action(state)
        observation, reward, done, _ = env.step(action)
        reward = torch.tensor([reward], device=device)
        if not done:
            next_state = observation
        else:
            next_state = None
        transition = Transition(state, action, next_state, reward)
        if exploit:
          priority = get_error([transition], policy_net, target_net).item()
        else:
          priority = abs(reward)
        # Store the transition in memory
        assert isinstance(transition, Transition)
        memory.push(priority, transition)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        loss = optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
        
        print(f'loss: {loss}', end='\r')
    print()
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print()
print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
