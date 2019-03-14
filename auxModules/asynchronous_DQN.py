import torch
import torch.nn as nn
import gym
import gym.spaces
import numpy as np
import torch.multiprocessing as mp
import collections
import cv2
from tensorboardX import SummaryWriter
import copy
import time

from multiprocessing import Process, Queue


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every 'skip'-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0], -1)
        return self.fc(conv_out)


def train(Q, QHat, device, rank, num_processes, frame_id, double, optimizer): #double is a boolen defining whether we want to use doudle-DQN or not
    env = make_env('PongNoFrameskip-v4')

    # Hyperparameters (mainly taken from Ch.6 of DRL Hands-on)
    nEpisode = 750
    GAMMA = 0.99
    EPSILON_0 = 1
    EPSILON_FINAL = 0.02
    DECAYING_RATE = 10 ** (-5)
    STORE_Q = 1000
    MAX_ITER = 200000
    BATCH_SIZE = 32
    REPLAY_SIZE = 10000
    REPLAY_START_SIZE = 10000

    epsilon = EPSILON_0
    buffer = collections.deque(maxlen=REPLAY_SIZE)
    loss_fn = torch.nn.MSELoss()
    total_rewards = []

    # visualize with tensorboardX
    writer = SummaryWriter(comment="-" + str(rank) + "-Pong")

    # best mean reward for the last 100 episodes
    best_mean_reward = None

    # main loop
    for step in range(nEpisode):
        print("process " + str(rank) + " is at episode " + str(step) + " out of " + str(nEpisode))
        obs = env.reset()
        total_reward = 0
        for _ in range(MAX_ITER):
            frame_id.value += 1
            local_frame_id = frame_id.value
            epsilon = max(EPSILON_FINAL, EPSILON_0 - local_frame_id * DECAYING_RATE)
            if np.random.random() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                obs1 = np.array([obs], copy=False)
                obs1 = torch.tensor(obs1).to(device)
                qVals = Q(obs1)
                _, actionV = torch.max(qVals, dim=1)
                action = int(actionV.item())
            obsNext, reward, done, _ = env.step(action)
            total_reward += reward
            buffer.append(collections.deque([obs, action, reward, done, obsNext]))
            obs = obsNext

            if len(buffer) >= REPLAY_START_SIZE:
                indices = np.random.choice(len(buffer), BATCH_SIZE, replace=False)
                observations, actions, rewards, dones, observationsNext = zip(*[buffer[idx] for idx in indices])

                observations, actions, rewards, dones, observationsNext = np.array(observations), np.array(
                    actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(
                    observationsNext)
                observationsV = torch.FloatTensor(observations).to(device)
                observationsNextV = torch.FloatTensor(observationsNext).to(device)
                actionsV = torch.tensor(actions).to(device)
                rewardsV = torch.tensor(rewards).to(device)
                doneMask = torch.ByteTensor(dones).to(device)
                # print(actionsV.shape)

                stateActionValues = Q(observationsV).gather(1, actionsV.unsqueeze(-1)).squeeze(-1)
                if double:
                    nextStateActions = Q(observationsNextV).max(1)[1]
                    nextStateValues = QHat(observationsNextV).gather(1, nextStateActions.unsqueeze(-1)).squeeze(-1)
                else:
                    nextStateValues = QHat(observationsNextV).max(1)[0]
                nextStateValues[doneMask] = 0.0
                nextStateValues = nextStateValues.detach()

                expectedStateActionValues = nextStateValues * GAMMA + rewardsV
                optimizer.zero_grad()
                loss = loss_fn(stateActionValues, expectedStateActionValues)
                loss.backward()
                optimizer.step()

            if local_frame_id % STORE_Q == 0:
                QHat = copy.deepcopy(Q)

            if done:
                break

        # report progress
        if total_reward is not None:
            total_rewards.append(total_reward)
            mean_reward = np.mean(total_rewards[-100:])
            writer.add_scalar("epsilon", epsilon, local_frame_id)
            writer.add_scalar("reward_100", mean_reward, local_frame_id)
            writer.add_scalar("reward", total_reward, local_frame_id)

        # save model and update best_mean_reward
        if (best_mean_reward is None or best_mean_reward < mean_reward) and len(buffer) >= REPLAY_START_SIZE:
            torch.save({
                'game': step,
                'model_state_dict': Q.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, "DQN_saved_models\\Pong_best.tar")
            best_mean_reward = mean_reward
        print(step, mean_reward, total_reward, local_frame_id)


if __name__ == "__main__":
    env_init = make_env('PongNoFrameskip-v4')
    start = time.time()
    mp.set_start_method('spawn')
    num_processes = 1
    double = False
    print("Using " + str(num_processes) + " processors\n")
    device = torch.device("cuda")
    Q = DQN(env_init.observation_space.shape, env_init.action_space.n).to(device)
    QHat = DQN(env_init.observation_space.shape, env_init.action_space.n).to(device)
    Q.share_memory()
    QHat.share_memory()
    LEARNING_RATE = 1e-4
    optimizer = torch.optim.Adam(Q.parameters(), lr=LEARNING_RATE)
    frame_id = mp.Value('i', 0)
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(Q, QHat, device, rank, num_processes, frame_id, double, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    end = time.time()
    print(end - start)
