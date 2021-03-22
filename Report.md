[//]: # (Image References)

[image1]: rewards.png "Plot Reward"

# Project 2: Continous Control

For solving the Tennis environment a Multi-Agent Deep Deterministic Policy Gradient algorithm was deployed. Basically, it consists of two agents each trained according to a centralized approach and executed in a decentralized manner. This means that the critic of an agent uses the whole observation space and action space (of all agents) while the actor of an agent chooses actions based only on its own state information.
Basically, there is a Multi Agent class which instantiates both single agents and which manages policy evaluation and policy improvement during training. Each agent whose methods and attributes are defined in class Agent behaves according to DDPG (with a target critic, local critic, target actor and local actor).

Each actor has the following properties:
It is composed of 3 Fully-Connected Layers.
* First Layer: input size = 24 output size = 256
* Second Layer: input size = 256 output size = 128
* Third Layer: input size = 128 output size = 4
The first and second layer use ReLu activation while the last one uses tanh activation.

The configuration of each critic looks as follows:
It is composed of 3 Fully-Connected Layers.
* First Layer: input size = 48 output size = 256
* Second Layer: input size = 260 output size = 128
* Third Layer: input size = 128 output size = 1
The first and second layer use ReLU activation.

Both agents share a common Replay Buffer to which they add their experiences each iteration. Moreover, random noise was added to the predicted actions during training. It was prefered over Ornstein-Uhlenbeck noise because it yields better results.
Actually, it wasn't that easy to train the algorithm because it acts in a very unstable manner. However, after a little bit of research and adjustments of hyperparameters the environment could be solved after 1262 episodes, with an average score of 0.5005 over the last 100 epochs (after taking the maximum over both agents).

![Plot Reward][image1]

The hyperparameters were chosen as follows:
BUFFER_SIZE = 100000  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3                 # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2       # how often to update the network (in timesteps)
UPDATE_PER_TIMESTEP=5  # how often to update the network per timestep
NOISE_DECAY = 0.99     # Noise reduction factor


In the future one could even improve performance by e.g. doing a hyperparameter optimization or by investigating other algorithms. Moreover, training time could certainly be decreased by using a parallel training procedure of  MADDPG (e.g. by using library multiprocessing). 