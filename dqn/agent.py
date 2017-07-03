from dqn.neural_net import NeuralNet
from dqn.replay_memory import ReplayMemory
from dqn.epsilon_greedy import EpsilonGreedy
from game_environments.pong.pong_game import Pong

import cv2

def main():
    try:
        W, H = 100, 100
        env = Pong(W, H)
        nn = NeuralNet(W,H, env.action_space['n'], 0.01, "pong")

        rm = ReplayMemory()
        epsilon_greedy = EpsilonGreedy()
        s = env.reset()
        action = env.sample()
        s1, r, t, _ = env.step(action)

        s  = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).reshape([100, 100, 1])
        s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY).reshape([100, 100, 1])

        rm.add((s,r,t,s1))
        batch = rm.get_minibatch()

        q = nn.Q([batch[i][0] for i in range(len(batch)) ])

        print(s1.shape)
        print(q)

    except KeyboardInterrupt:
        nn.close_session()

main()
