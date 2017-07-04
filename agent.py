"""
Author: Uriel Sade
Date: July 4rd, 2017
"""

from dqn.neural_net import NeuralNet
from dqn.replay_memory import ReplayMemory
from dqn.epsilon_greedy import EpsilonGreedy
from game_environments.pong.pong_game import Pong
from game_environments.snake.snake_game import SnakeGame

import numpy as np
import cv2

def main():

    training = True

    try:
        W, H = 100, 100

        env = SnakeGame(10,10, training=training)
        #env = Pong(W, H)
        nn = NeuralNet(W,H, env.action_space['n'], env.GAME_TITLE, gamma=0.97, learning_rate=0.0001)

        replay_memory = ReplayMemory(capacity=50000)
        #epsilon_greedy = EpsilonGreedy(initial_value=0.1, fixed=True)
        epsilon_greedy = EpsilonGreedy()

        s = env.reset()
        s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).reshape([W, H, 1])
        max_score = 0
        games_played = 0
        scores = {}
        while True:
            # make 10 moves, then train on a minibatch
            for i in range(10):
                take_random = epsilon_greedy.evaluate()
                if training and take_random:
                    a = env.sample()
                else:
                    a = nn.predict([s])[0]
                s1, r, t, score = env.step(a)
                s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY).reshape([W, H, 1])
                replay_memory.add((s, a, r, s1, t))

                if not t:
                    s = s1
                else:
                    max_score = max(max_score, score)
                    games_played += 1
                    scores[score] = scores.get(score, 0) + 1
                    print("\rMax Score: {:3} || Score: {:3} || Games Played: {:10} Epsilon: {:.5f} Scores: ".format(max_score, score, games_played, epsilon_greedy.peek()), scores,  end="")
                    s = env.reset()
                    s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).reshape([W, H, 1])
            if training:
                batch = replay_memory.get_minibatch()
                loss = nn.optimize(batch)

    except KeyboardInterrupt:
        if training:
            nn.save()

        nn.close_session()

main()
