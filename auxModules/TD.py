import gym
import numpy as np

env = gym.make("FrozenLake8x8-v0")

def chooseAction(q_table, state, epsilon=0, softmax=True, tau=0.01):
    """Choose an action.

  Args:
    q_table: Q-table.
    state: state
    epsilon: epsilon
    softmax: boolean set to $True$ if softmax exploration is used
    tau: parameter of the softmax exploration.
  Returns:
    Action
  """
    if softmax:
        m = max(q_table[state])
        aux = np.exp((q_table[state] - m) / tau)
        d = np.sum(aux)
        return np.random.choice(np.arange(env.action_space.n), p=aux / d)
    if np.random.random() > epsilon:
        return np.argmax(q_table[state])
    return env.action_space.sample()


def routine(algo, nEpisode=2000, gamma=0.99, alpha=0.4, epsilon0=0.9, epsilonMin=0.05, decreaseRate=0.999,
            softmax=False, tau=0.01, window=100):
    """Body function used to implement both $SARSA$ and $Q$-Learning, since only the estimation of the error changes between the two methods.

  Args:
    algo: either "SARSA" or "QLearning"
* nEpisode: number of episodes we simulate to update our Q-function
* gamma: discount factor
* alpha: learning rate
* epsilon_0: initial value of epsilon
* epsilonMin: minimal value of $\epsilon$ if a decaying $\epsilon$ is used
* decreaseRate: geometric parameter of the decaying of epsilon (set to 1 if no decaying)
* softmax: boolean set to True if softmax exploration is used
* tau: parameter of the softmax exploration
* window: number of episodes we consider to compute the success rate (or accuracy) of the Q-function ; ratio of successes on the last $window$ episodes
  Returns:
    q_table: the Q-table
    histAcc: array where accuracies are stored
  """
    accuracy = 0
    epsilon = epsilon0
    q_table = np.ones((env.observation_space.n, env.action_space.n))
    result = np.zeros(window)
    histAcc = [0]
    episode = 0

    for ep in range(nEpisode):
        epsilon = max(epsilonMin, decreaseRate * epsilon)
        observation0 = env.reset()
        action0 = chooseAction(q_table, observation0, epsilon, softmax, tau)
        done = False
        for H in range(200):
            observation1, reward, done, info = env.step(action0)
            action1 = chooseAction(q_table, observation1, epsilon, softmax, tau)
            err = reward - q_table[observation0, action0]

            if not done:
                if algo == "SARSA":
                    err += gamma * q_table[observation1, action1]
                if algo == "QLearning":
                    err += gamma * q_table[observation1, np.argmax(q_table[observation1])]

            q_table[observation0, action0] += alpha * err

            if done:
                break

            observation0, action0 = observation1, action1

        success = 1 if (reward == 1) else 0
        accuracy += (success - result[episode]) / window
        result[episode] = success
        episode = (episode + 1) % window
        if episode == 0 or episode == window // 2:  # accuracy ratio is stored once in @window/2 episodes
            histAcc.append(accuracy)

    return q_table, histAcc


def routineTh(algo, threshold=0.8, nEpisodeMax = 30000, gamma=0.99, alpha=0.4, epsilon0=0.9, epsilonMin=0.05, decreaseRate=0.999,
              softmax=False, tau=0.01, window=100):
    accuracy = 0
    epsilon = epsilon0
    q_table = np.ones((env.observation_space.n, env.action_space.n))
    result = np.zeros(window)
    histAcc = [0]
    episode = 0

    while accuracy < threshold and episode < nEpisodeMax:
        epsilon = max(epsilonMin, decreaseRate * epsilon)
        observation0 = env.reset()
        action0 = chooseAction(q_table, observation0, epsilon, softmax, tau)
        done = False
        for H in range(200):
            observation1, reward, done, info = env.step(action0)
            action1 = chooseAction(q_table, observation1, epsilon, softmax, tau)
            err = reward - q_table[observation0, action0]

            if not done:
                if algo == "SARSA":
                    err += gamma * q_table[observation1, action1]
                if algo == "QLearning":
                    err += gamma * q_table[observation1, np.argmax(q_table[observation1])]

            q_table[observation0, action0] += alpha * err

            if done:
                break

            observation0, action0 = observation1, action1

        accuracy += (reward - result[episode % window]) / window
        result[episode % window] = reward
        episode = episode + 1
        histAcc.append(accuracy)

    return q_table, histAcc, episode

def SARSA(nEpisode = 2000, gamma = 0.999, alpha = 0.4, epsilon0 = 0.9, epsilonMin = 0.05, decreaseRate = False, softmax = True, tau = 0.01, window = 100):
    return routine("SARSA", nEpisode, gamma, alpha, epsilon0, epsilonMin, decreaseRate, softmax, tau, window)

def SARSATh(threshold = 0.8, nEpisodeMax = 30000, gamma = 0.999, alpha = 0.4, epsilon0 = 0.9, epsilonMin = 0.05, decreaseRate = False, softmax = True, tau = 0.01, window = 100):
    return routineTh("SARSA", threshold, nEpisodeMax, gamma, alpha, epsilon0, epsilonMin, decreaseRate, softmax, tau, window)

def QLearning(nEpisode = 2000, gamma = 0.999, alpha = 0.4, epsilon0 = 0.9, epsilonMin = 0.05, decreaseRate = True, softmax = False, tau = 0.003, window = 100):
    return routine("QLearning", nEpisode, gamma, alpha, epsilon0, epsilonMin, decreaseRate, softmax, tau, window)

def QLearningTh(threshold = 0.8, nEpisodeMax = 30000, gamma = 0.999, alpha = 0.4, epsilon0 = 0.9, epsilonMin = 0.05, decreaseRate = False, softmax = True, tau = 0.01, window = 100):
    return routineTh("QLearning", threshold, nEpisodeMax, gamma, alpha, epsilon0, epsilonMin, decreaseRate, softmax, tau, window)

def testPolicy (q_table, nEpisode = 2000):
    success = 0
    for _ in range(nEpisode):
        t = 0
        observation = env.reset()
        done  = False
        actionTable = np.argmax(q_table, axis = 1)
        while not done and t < 200:
            action = actionTable[observation]
            observation, reward, done, info = env.step(action)
            t += 1

        if reward == 1:
            success += 1
    return success / nEpisode