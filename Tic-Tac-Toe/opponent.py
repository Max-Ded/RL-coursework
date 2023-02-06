from util_func import *
import random

class Opponent:
    """
    BOT Player than places his symbol on the grid based on his strategy function
    """
    def __init__(self,strategy):

        self.strategy = strategy

    def play(self,state,symbol):
        return self.strategy(state,symbol)