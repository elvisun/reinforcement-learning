import pygame
import random

#TODO: map ball's y_speed to impact position along the paddles

class Pong:

    """
    param W: The width  of the game's window
    param H: The height of the game's window
    """
    def __init__(self, W, H):
        self.WINDOW_WIDTH  = W
        self.WINDOW_HEIGHT = H
        self.PADDLE_WIDTH  = W//50
        self.PADDLE_HEIGHT = H//10

        # TODO: Change to radius and make the ball a circle
        self.BALL_WIDTH = W//40

        self.PADDLE_SENSITIVITY = max(1, self.PADDLE_HEIGHT//5)
        self.BALL_SPEED = max(1, self.BALL_WIDTH//3)

        self.COLOR_BLACK = (0,0,0)
        self.COLOR_WHITE = (255,255,255)
        self.COLOR_BLUE = (0,0,255)
        self.COLOR_RED = (255,0,0)

        self.GAME_TITLE = 'Pong'
        self.world = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.action_space = {'n':2, 'ACTION_CODES':[0,1], 'ACTIONS':['DOWN', 'UP']}

        pygame.display.set_caption(self.GAME_TITLE)

        self.reset()

    """
    Performs an action in the enviromnent causing a transition to the next state
    return: (s, r, t, sc)
        s -> new state (ndarray)
        r -> Reward(state, action) (float)
        t -> is this a terminal state, i.e. won or lost (boolean)
        sc-> the running score (# times the ball was successfully blocked)
    """
    def step(self, action):
        pygame.event.get() # flushes the event queue
        self.world.fill(self.COLOR_BLACK)
        self.move_player_paddle(action)
        self.move_opponent_paddle()
        self.move_ball()
        reward, terminal = self.check_status()
        self.draw()
        return (self.get_state(), reward, terminal, self.score)

    """
    Resets the game environment
    return: The initial state (ndarray)
    """
    def reset(self):
        self.player_paddle = {
                               'x':self.WINDOW_WIDTH//10,
                               'y':self.WINDOW_HEIGHT//2 - self.PADDLE_HEIGHT//2
                            }
        self.opponent_paddle = {
                               'x':self.WINDOW_WIDTH - self.player_paddle['x'],
                               'y':self.player_paddle['y']
                            }
        self.ball = {
                      'x': self.WINDOW_WIDTH//2 - self.BALL_WIDTH//2,
                      'y': self.WINDOW_HEIGHT//2 - self.BALL_WIDTH//2,
                      'x_direction': -1,
                      'y_direction': random.sample([-1,1], 1)[0]
                     }
        self.score = 0
        self.draw()
        return self.get_state() # game's inital state

    """
    Moves the player's paddle either up or down according to the action
    param action: the action being performed
    raises ValueError: if the action is invalid
    """
    def move_player_paddle(self, action):
        if action == 0 and self.player_paddle['y'] + self.PADDLE_HEIGHT < self.WINDOW_HEIGHT:
            self.player_paddle['y'] += self.PADDLE_SENSITIVITY
        elif action == 1 and self.player_paddle['y'] > 0:
            self.player_paddle['y'] -= self.PADDLE_SENSITIVITY

        if action not in self.action_space['ACTION_CODES']:
            raise ValueError('Error: Invalid action: {}'.format(action))
        return

    """
    Moves the opponent's paddle in the diraction of the ball
    """
    def move_opponent_paddle(self):
        distance =  self.ball['y'] - self.opponent_paddle['y']

        if distance > 0 and self.opponent_paddle['y'] + self.PADDLE_HEIGHT < self.WINDOW_HEIGHT:
            self.opponent_paddle['y'] += min(self.PADDLE_SENSITIVITY, abs(distance))
        elif self.opponent_paddle['y'] > 0:
            self.opponent_paddle['y'] -= min(self.PADDLE_SENSITIVITY, abs(distance))
        return

    """
    Moves the ball according to its direction vector
    """
    def move_ball(self):

        self.ball['y'] += self.ball['y_direction']*self.BALL_SPEED
        self.ball['x'] += self.ball['x_direction']*self.BALL_SPEED

        if self.ball['y'] < 0:
            self.ball['y'] = 0
            self.ball['y_direction'] = 1
        elif self.ball['y'] + self.BALL_WIDTH > self.WINDOW_HEIGHT:
            self.ball['y'] = self.WINDOW_HEIGHT - self.BALL_WIDTH
            self.ball['y_direction'] = -1

        return
    """
    Checks the game's status and updates the ball's direction if necessary
    return: (reward, terminal)
        reward   -> the reward from the last action
        terminal -> is this a terminal state, i.e. won or lost (boolean)
    """
    def check_status(self):
        reward = 0
        ball_midpoint_x = self.ball['x'] + self.BALL_WIDTH//2
        player_paddle_right = self.player_paddle['x'] + self.PADDLE_WIDTH
        opponent_paddle_left = self.opponent_paddle['x']

        if self.ball['x'] <= 0:
            return (-1, True)
        elif self.ball['x'] <= player_paddle_right \
            and self.intersects_ball(self.player_paddle):
            #if self.ball['x_direction'] = -1: reward = 1
            self.ball['x_direction'] = 1
            reward = 1
            self.score += 1

        if ball_midpoint_x > opponent_paddle_left:
            return (0, True)
        elif self.ball['x'] + self.BALL_WIDTH >= opponent_paddle_left \
            and self.intersects_ball(self.opponent_paddle):
            self.ball['x_direction'] = -1

        return (reward, False)

    """
    return: true iff the paddle intersects the ball
    """
    def intersects_ball(self, paddle):
        ball_l = self.ball['x']
        ball_r = ball_l + self.BALL_WIDTH
        ball_t = self.ball['y']
        ball_b = ball_t + self.BALL_WIDTH
        paddle_l = paddle['x']
        paddle_r = paddle_l + self.PADDLE_WIDTH
        paddle_t = paddle['y']
        paddle_b = paddle_t + self.PADDLE_HEIGHT

        return ball_l <= paddle_r and ball_r > paddle_l \
            and ball_b > paddle_t and ball_t < paddle_b


    """
    return: the raw pixel data of the game (ndarray)
    """
    def get_state(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())

    """
    Draws the paddles and ball
    """
    def draw(self):
        self.draw_rect( self.player_paddle['x'], self.player_paddle['y'],
                        self.PADDLE_WIDTH, self.PADDLE_HEIGHT, self.COLOR_WHITE
                        )
        self.draw_rect( self.opponent_paddle['x'], self.opponent_paddle['y'],
                        self.PADDLE_WIDTH, self.PADDLE_HEIGHT, self.COLOR_RED
                        )
        self.draw_rect( self.ball['x'], self.ball['y'],
                        self.BALL_WIDTH, self.BALL_WIDTH, self.COLOR_BLUE
                        )
        pygame.display.flip()
        return

    """
    Helper function to draw rectangles on the screen
    param x: The upper side of the rectangle
    param y: The left  side of the rectangle
    param W: The width of the rectangle (extends to the right of x)
    param H: The height of the rectangle (extends to the left of y)
    param color: an RGB tuple representing the rectangle's color
    """
    def draw_rect(self, x, y, W, H, color):
        rect = pygame.Rect( x, y, W, H )
        pygame.draw.rect(self.world, color, rect, 0)
        return
