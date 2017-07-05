"""
Author: Uriel Sade
Date: July 4rd, 2017
"""

#TODO: move this into a class and run it's methods from a main script

from dqn.neural_net import NeuralNet
from dqn.replay_memory import ReplayMemory
from dqn.epsilon_greedy import EpsilonGreedy
from game_environments.pong.pong_game import Pong
from game_environments.snake.snake_game import SnakeGame
import util.parser as parser
import util.stats_saver as stats_saver

import numpy as np
import cv2
import time

def main():

    opt, args = parser.get_arguments()
    training = parser.str2bool(opt.training)
    start_time = time.time()

    max_score = 0
    games_played = 0
    frame_iterations = 0
    scores = {}

    print("Training: ", training)

    try:
        W, H = 100, 100

        env = SnakeGame(10,10, training=training)
        #env = Pong(W, H)
        nn = NeuralNet(W,H, env.action_space['n'], env.GAME_TITLE, gamma=0.97, learning_rate=0.0001)

        replay_memory = ReplayMemory(capacity=50000)
        epsilon_greedy = EpsilonGreedy(initial_value=0.1, target_value=0.03, exploration_frames=1e4)
        #epsilon_greedy = EpsilonGreedy()

        s = env.reset()
        s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).reshape([W, H, 1])
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
                frame_iterations+=1

                if not t:
                    s = s1
                else:
                    max_score = max(max_score, score)
                    games_played += 1
                    scores[score] = scores.get(score, 0) + 1
                    e_value = 0 if not training else epsilon_greedy.peek()
                    print("\rMax Score: {:3} || Score: {:3} || Games Played: {:10} Epsilon: {:.5f} Scores: {}".format(max_score, score, games_played, e_value, str(scores)),  end="")
                    s = env.reset()
                    s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).reshape([W, H, 1])
            if training:
                batch = replay_memory.get_minibatch()
                loss = nn.optimize(batch)

    except KeyboardInterrupt:
        if training:
            nn.save()
            print("\nCheckpoint saved")
        nn.close_session()
        stats_saver.save_to_file(max_score, games_played, frame_iterations, scores, training, start_time)
        print("Session closed")

main()
