from util_func import *
import random

def choice_at_random(state,symbol):
    """
    Given a state and symbol, plays at random on the grid and returns the resulting state
    """
    empty_spaces = [i for i in range(len(state)) if state[i]=="-"]
    random.shuffle(empty_spaces)
    if len(empty_spaces)==0:
        return state
    return replace_in_str(state,empty_spaces[0],symbol)

def choice_attack_defense(state,symbol):
    """
    Given a state and symbol, defends if it is about to loose, attack if it can win and else plays randomly
    """
    not_symbol = "O" if symbol=="X" else "X"
    empty_spaces = [i for i in range(len(state)) if state[i]=="-"]
    random.shuffle(empty_spaces)
    for space in empty_spaces:
        temp = replace_in_str(state,space,symbol=not_symbol)
        if has_won(temp,not_symbol):
            #if the state is winning for the opponent
            return replace_in_str(state,space,symbol=symbol)
    return choice_immediate_best(state,symbol=symbol)

def choice_immediate_best(state,symbol):
    """
    Given a state and a symbol, will play at random except if there is a winning move
    """
    not_symbol = "O" if symbol=="X" else "X"
    empty_spaces = [i for i in range(len(state)) if state[i]=="-"]
    random.shuffle(empty_spaces)
    for space in empty_spaces:
        temp = replace_in_str(state,space,symbol=symbol)
        if has_won(temp,symbol=symbol):
            return temp
    return choice_at_random(state,symbol=symbol)

class Opponent:
    """
    BOT Player than places his symbol on the grid based on his strategy function
    """
    def __init__(self,strategy):

        self.strategy = strategy

    def play(self,state,symbol):
        return self.strategy(state,symbol)