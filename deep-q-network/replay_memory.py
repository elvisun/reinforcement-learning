import numpy as np

"""
Stores (s_t1, a_t1, r_t1, s_t2) tuples
    s_t1 -> State at time t1
    a_t1 -> Action taken at time t1
    r_t1 -> Reward received at time t1
    s_t2 -> The resulting state at time (t1 + 1) (i.e. t2)
"""
class ReplayMemory:

    """
    param capacity: The maximum entry capacity of the replay memory
    param batch_size: The desired size of each sampled minibatch
    """
    def __init__(self, capacity=50000, batch_size=32):

        DEFAULT_CAPACITY = 50000
        self.MAX_CAPACITY = capacity if capacity >= 1000 else DEFAULT_CAPACITY
        self.MINIBATCH_SIZE = batch_size
        self.internal_memory = []
        self.replace_index = 0
    """
    Adds an entry to the replay memory
    param addition: A tuple containing (s_t1, a_t1, r_t1, s_t2)
    """
    def add(self, addition):
        if len(self.internal_memory) < self.MAX_CAPACITY:
            self.internal_memory.append(addition)
        else:
            self.internal_memory[self.replace_index] = addition
            self.replace_index = (self.replace_index + 1) % self.MAX_CAPACITY

    """
    param batch_size (optional): the desired size of the minibatch
    return: A list from the replay memory containing
            random (s_t1, a_t1, r_t1, s_t2) tuples
    """
    def get_minibatch(self, batch_size=None):
        if not batch_size: batch_size = self.MINIBATCH_SIZE

        indices = np.random.choice( len(self.internal_memory),
                                    batch_size, replace=False )

        return [self.internal_memory[i] for i in indices]
