""" Script that was used to initially test the pong enviroment as a whole """

from pong import Pong
import random

env = Pong(600, 400)

while True:

    env.reset()
    t = False
    i = 0
    while not t:
        # random
        #action = 0 if random.random() < 0.5 else 1
        # periodic
        action = 0 if i > 100 else 1

        if i == 200:
            i = 0
        i += 1
        s, r, t, sc = env.step(action)

        print("Reward: {:2} Action: {:2} Terminal: {:5} Score: {:3}".format(r, env.action_space["ACTIONS"][action], t, sc))
