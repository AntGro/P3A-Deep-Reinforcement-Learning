# P3A-Deep-Reinforcement-Learning
Research project focusing on Deep Reinforcement learning.

## TD Learning
In auxModules package are implementations of SARSA and Q-Learning algorithms with several exploration methods (including Ɛ-greedy and softmax).
In Notebooks/TD_Learning.py we compared perfomance of these methods applied on deterministic and stochastic Frozen-Lake environment.


## Deep Q-Learning
In DQL file:
* DQN.py: implementation of Deep Q-Network algorithm on Pong-v0 environment. Modifying some parameters activate speed-up techniques such as Double DQN and N-steps DQN.
* asynchronous_learning.py: implementation of a basic asynchronous (multithread) DRL algorithm inspired by https://arxiv.org/pdf/1602.01783.pdf.
* asynchronous_DQN.py: implementation of an asynchronous (multithread) DQN algorithm.
