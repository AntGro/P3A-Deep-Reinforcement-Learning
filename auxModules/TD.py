import numpy as np
import pandas as pd
from tqdm import tqdm
import gym

def chooseAction(env, q_table, state, epsilon=0, softmax=True, tau=0.01):
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


def routine(env, algo, nEpisode=2000, gamma=0.99, alpha=0.4, epsilon0=0.9, epsilonMin=0.05, decreaseRate=0.999,
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
        action0 = chooseAction(env, q_table, observation0, epsilon, softmax, tau)
        done = False
        for H in range(200):
            observation1, reward, done, info = env.step(action0)
            action1 = chooseAction(env, q_table, observation1, epsilon, softmax, tau)
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
        histAcc.append(accuracy)

    return q_table, histAcc


def routineTh(env, algo, threshold=0.8, nEpisodeMax=30000, gamma=0.99, alpha=0.4, epsilon0=0.9, epsilonMin=0.05,
              decreaseRate=0.999,
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
        action0 = chooseAction(env, q_table, observation0, epsilon, softmax, tau)
        done = False
        for H in range(200):
            observation1, reward, done, info = env.step(action0)
            action1 = chooseAction(env, q_table, observation1, epsilon, softmax, tau)
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


def SARSA(env, nEpisode=2000, gamma=0.999, alpha=0.4, epsilon0=0.9, epsilonMin=0.05, decreaseRate=1, softmax=True,
          tau=0.01, window=100):
    return routine(env, "SARSA", nEpisode, gamma, alpha, epsilon0, epsilonMin, decreaseRate, softmax, tau, window)


def SARSATh(env, threshold=0.8, nEpisodeMax=40000, gamma=0.999, alpha=0.4, epsilon0=0.9, epsilonMin=0.05,
            decreaseRate=1, softmax=True, tau=0.01, window=100):
    return routineTh(env, "SARSA", threshold, nEpisodeMax, gamma, alpha, epsilon0, epsilonMin, decreaseRate, softmax,
                     tau, window)


def QLearning(env, nEpisode=2000, gamma=0.999, alpha=0.4, epsilon0=0.9, epsilonMin=0.05, decreaseRate=1,
              softmax=False, tau=0.003, window=100):
    return routine(env, "QLearning", nEpisode, gamma, alpha, epsilon0, epsilonMin, decreaseRate, softmax, tau, window)


def QLearningTh(env, threshold=0.8, nEpisodeMax=4000, gamma=0.999, alpha=0.4, epsilon0=0.9, epsilonMin=0.05,
                decreaseRate=1, softmax=True, tau=0.01, window=100):
    return routineTh(env, "QLearning", threshold, nEpisodeMax, gamma, alpha, epsilon0, epsilonMin, decreaseRate,
                     softmax, tau, window)


def testPolicy(env, q_table, nEpisode=2000):
    success = 0
    for _ in range(nEpisode):
        t = 0
        observation = env.reset()
        done = False
        actionTable = np.argmax(q_table, axis=1)
        while not done and t < 200:
            action = actionTable[observation]
            observation, reward, done, info = env.step(action)
            t += 1

        if reward == 1:
            success += 1
    return success / nEpisode


def compareMethods(envName, nEpisodeAccuracy=100, threshold=0.8, nEpisodeMax=2000, nIter=5,
                   Eps=0.1 * np.arange(1, 10, 2),
                   DR=[0.9, 0.99, 0.999], eps=0.9, Tau=[1, 0.1, 0.01, 0.001]):
    env = gym.make(envName)

    recap = pd.DataFrame(
        columns=["Accuracy - SARSA", "Nb episodes - SARSA", "Accuracy - QLearning", "Nb episodes - QLearning"])

    ## epsilon-greedy with fixed epsilon
    print("epsilon-greedy with fixed epsilon")
    for eps in tqdm(Eps):
        sarsa = 0
        sarsaEpisode = 0
        ql = 0
        qEpisode = 0
        for _ in range(nIter):
            q_table, histAcc, qEp = QLearningTh(env, threshold, nEpisodeMax=nEpisodeMax, epsilon0=eps, decreaseRate=1)
            ql += testPolicy(env, q_table, nEpisodeAccuracy)
            qEpisode += qEp
            q_table, histAcc, sarsaEp = SARSATh(env, threshold, nEpisodeMax=nEpisodeMax, epsilon0=eps, decreaseRate=1)
            sarsa += testPolicy(env, q_table, nEpisodeAccuracy)
            sarsaEpisode += sarsaEp
        sarsa /= nIter
        sarsaEpisode /= nIter
        qEpisode /= nIter
        ql /= nIter
        recap.loc["Fixed $\epsilon$ : $\epsilon$ = {}".format(round(eps, 2))] = [sarsa, sarsaEpisode, ql, qEpisode]

    ## epsilon-greedy with decreasing epsilon
    print("epsilon-greedy with decreasing epsilon")
    for dr in tqdm(DR):
        sarsa = 0
        sarsaEpisode = 0
        ql = 0
        qEpisode = 0
        for _ in range(nIter):
            q_table, histAcc, qEp = QLearningTh(env, threshold, epsilon0=eps, nEpisodeMax=nEpisodeMax, decreaseRate=dr)
            ql += testPolicy(env, q_table, nEpisodeAccuracy)
            qEpisode += qEp
            q_table, histAcc, sarsaEp = SARSATh(env, threshold, epsilon0=eps, nEpisodeMax=nEpisodeMax, decreaseRate=dr)
            sarsa += testPolicy(env, q_table, nEpisodeAccuracy)
            sarsaEpisode += sarsaEp
        sarsa /= nIter
        sarsaEpisode /= nIter
        qEpisode /= nIter
        ql /= nIter
        recap.loc["Decaying-$\epsilon$ : decaying rate = {}".format(dr)] = [sarsa, sarsaEpisode, ql, qEpisode]

    ##Softmax
    print("Softmax")
    for t in tqdm(Tau):
        sarsa = 0
        sarsaEpisode = 0
        ql = 0
        qEpisode = 0
        for _ in range(nIter):
            q_table, histAcc, qEp = QLearningTh(env, threshold, nEpisodeMax=nEpisodeMax, softmax=True, tau=t)
            ql += testPolicy(env, q_table, nEpisodeAccuracy)
            qEpisode += qEp
            q_table, histAcc, sarsaEp = SARSATh(env, threshold, nEpisodeMax=nEpisodeMax, softmax=True, tau=t)
            sarsa += testPolicy(env, q_table, nEpisodeAccuracy)
            sarsaEpisode += sarsaEp
        sarsa /= nIter
        sarsaEpisode /= nIter
        qEpisode /= nIter
        ql /= nIter
        recap.loc["Softmax : $\tau$ = {}".format(t)] = [sarsa, sarsaEpisode, ql, qEpisode]

    # Save recap
    recap.to_csv("output/" + envName + ".csv")

