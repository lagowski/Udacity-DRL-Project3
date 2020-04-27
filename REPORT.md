## Deep Reinforcement Learning - Project 3: Collaboration & Competition

### Description

The environment is created by multi-agent Deep Deterministic Policy Gradient (DDPG) algorithm. Training proceeds as follows:

#### Background for Deep Deterministic Policy Gradient (DDPG)
MADDPG find its origins in an off-policy method called **Deep Deterministic Policy Gradient (DDPG)** and described in the paper Continuous control with deep reinforcement learning.

> We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies end-to-end: directly from raw pixel inputs.

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

More details available on the Open AI's [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) website.

![DDPG algorithm from Spinning Up website](./images/DDPG.svg)

This algorithm screenshot is taken from the [DDPG algorithm from the Spinning Up website](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

#### Multi Agent Deep Deterministic Policy Gradient (MADDPG)



#### In this case the workflow is

- The 2 agents each receive a different state vector (with 24 elements) from the environment
- Each agent feeds its state vector through the local actor network to get an action vector (with 2 elements) as output. Noise based on an Ornstein-Uhlenbeck process is added to the predicted actions to encourage exploration
- Each agent then receives a next state vector and a reward from the environment (as well as a termination signal that indicates if the episode is complete)
- The experience tuple `(state, action, reward, next state)` of each agent is added to a common replay buffer
- A random sample of experience tuples is drawn from the replay buffer (once it contains enough) 
- The sample is used to update the weights of the local critic network:
    1. The next state is fed into the target actor to obtain the next action
    1. The (next state, next action) pair is fed into the target critic to obtain an action value, Q_next
    1. The action value for the current state is then computed as Q_cur = reward + gamma*Q_next
    1. The (current state, current action) pair are fed into the local critic to obtain a predicted action value, Q_pred
    1. The MSE loss is computed between Q_cur and Q_pred, and the weights of the local critic are updated accordingly
- The sample is used to update the weights of the local actor network:
    1. The current state is fed into the local actor to obtain predicted a action
    1. Each (current state, predicted action) pair for the sample is fed into the local critic to obtain action values
    1. The negative mean of the predicted Q values is used as a loss function to update the weights of the local actor
- The target actor and critic networks are then soft-updated by adjusting their weights slightly toward those of their local counterparts
- The states that were obtained in step (3) then become the current states for each agent and the process repeats from step (2)

#### Learning Algorithms

This project considers **Multi Agent Deep Deterministic Policy Gradient (MADDPG)** which is  described in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)

> We explore deep reinforcement learning methods for multi-agent domains. We begin by analyzing the difficulty of traditional algorithms in the multi-agent case: Q-learning is challenged by an inherent non-stationarity of the environment, while policy gradient suffers from a variance that increases as the number of agents grows. We then present an adaptation of actor-critic methods that considers action policies of other agents and is able to successfully learn policies that require complex multi-agent coordination. Additionally, we introduce a training regimen utilizing an ensemble of policies for each agent that leads to more robust multi-agent policies. We show the strength of our approach compared to existing methods in cooperative as well as competitive scenarios, where agent populations are able to discover various physical and informational coordination strategies.

![MADDPG algorithm](./images/MADDPG-algo.png) (screenshot from the paper)

The main concept behind this algorithm is summarized in this illustration taken from the paper :

![Overview of the multi-agent decentralized actor, centralized critic approach](./images/MADDPG.png) (screenshot from the paper)

> we accomplish our goal by adopting the framework of centralized training with
decentralized execution. Thus, we allow the policies to use extra information to ease training, so
long as this information is not used at test time. It is unnatural to do this with Q-learning, as the Q
function generally cannot contain different information at training and test time. Thus, we propose
a simple extension of actor-critic policy gradient methods where the critic is augmented with extra
information about the policies of other agents.

In short, this means that during the training, the Critics networks have access to the states and actions information of both agents, while the Actors networks have only access to the information corresponding to their local agent.


#### Agent Hyperparameters

- `GAMMA = 0.99`
- `TAU = 0.001` 
- `LR_ACTOR = 0.001` 
- `LR_CRITIC = 0.001`
- `BUFFER_SIZE = 100000` 
- `BATCH_SIZE = 256` 
- `theta = 0.15` `sigma = 0.05` 


#### Network Architectures and Hyperparameters

The actor network takes a state vector (24 elements) as input and returns an action vector (2 elements). It was modelled with a feedforward deep neural network comprising a 24 dimensional input layer, two hidden layers with 128 neurons and ReLU activations and a 2 dimensional output layer with a tanh activation to ensure that the predicted actions are in the range -1 to +1. Batch normalisation was applied to the input and two hidden layers. 

The critic network takes the state and action vectors as input, and returns a scalar Q value as output. It was modelled with a feedforward deep neural network with a 24 dimensional input layer (for the state vector) that was fully connected to 128 neurons in the first hidden layer with ReLU activations. The outputs of the first layer were batch normalised and concatenated with the 2 dimensional action vector as input to the second hidden layer, which also comprised 128 neurons with ReLU activations. Finally, the second hidden layer mapped to an output layer with single neuron and linear activation (outputs a single real number). 


### Results

![results.png](/assets/result.png)


### Future Plans for Improvement

- **Hyperparameter tuning** - I focused on tuning hidden size and add random exploration. Other parameters might be also good candidates and speed up the training.
- **MAPPO** -  I started this project with my own adaptation of PPO for multi agent. It learned something, but wasn't nearly as good as MADDPG. I would like to make it work
- **Try it on other environment** - Soccer environment sounds like a good idea to tackle.

