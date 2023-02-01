import numpy as np
import random

grid = np.zeros((3,3),dtype=int)
grid[1,1]=1
grid[1,2]=2

def encode_state(grid:list):
    """
    Takes grid as 3x3 Array => Returns the char[9] representation 
    """
    return "".join(["".join(x) for x in grid])
def decode_state(grid:str):
    """
    Takes the char[9] representation => return grid as 3x3 Array
    """
    return [list(grid[:3]),list(grid[3:6]),list(grid[6:])]
def pretty_print_grid(grid):
    """
    Takes the grid (list or string representation) and prints it in the console (if string will convert to list before printing)
    """
    if type(grid)==str:
        pretty_print_grid(decode_state(grid))
    else:
        print("|".join(grid[0]))
        print("|".join(grid[1]))
        print("|".join(grid[2]),end="\n\n")


#All Winning state for "X"/"O", replace the other symbols "-" and "O"/"X" before use
winning_state = { "X":{
    "XXX&&&&&&":1,
    "&&&XXX&&&":1,
    "&&&&&&XXX":1,
    "X&&&X&&&X":1,
    "&&X&X&X&&":1,
    "X&&X&&X&&":1,
    "&X&&X&&X&":1,
    "&&X&&X&&X":1 },
                "O": {
    "OOO&&&&&&":1,
    "&&&OOO&&&":1,
    "&&&&&&OOO":1,
    "O&&&O&&&O":1,
    "&&O&O&O&&":1,
    "O&&O&&O&&":1,
    "&O&&O&&O&":1,
    "&&O&&O&&O":1 }
}

#Winning states for "X"/"O" as a list
winning_state_list = {k:list(v.keys()) for k,v in winning_state.items()}

def type_conversion(func):
    '''Decorator that convert list input to string (3x3 matrix are flattened and "".joined)'''
    def wrap(state):
        if type(state)==list:
            result = decode_state(func(encode_state(state)))
        else:
            result = func(state)
        return result
    return wrap
  
"""
Perfoms the 7 tranformation on the grid that conserve winning states 
"""
@type_conversion
def rotate_270(state):
    return f"{state[2]}{state[5]}{state[8]}{state[1]}{state[4]}{state[7]}{state[0]}{state[3]}{state[6]}"
@type_conversion
def rotate_180(state):
    return f"{state[8]}{state[7]}{state[6]}{state[5]}{state[4]}{state[3]}{state[2]}{state[1]}{state[0]}"
@type_conversion
def rotate_90(state):
    return f"{state[6]}{state[3]}{state[0]}{state[4]}{state[7]}{state[1]}{state[8]}{state[5]}{state[2]}"
@type_conversion
def mirror_diag_2(state):
    return f"{state[8]}{state[5]}{state[2]}{state[7]}{state[4]}{state[1]}{state[6]}{state[3]}{state[0]}"
@type_conversion
def mirror_diag_1(state):
    return f"{state[0]}{state[3]}{state[6]}{state[1]}{state[4]}{state[7]}{state[2]}{state[5]}{state[8]}"
@type_conversion
def mirror_y(state):
    return f"{state[2]}{state[1]}{state[0]}{state[5]}{state[4]}{state[3]}{state[8]}{state[7]}{state[6]}"
@type_conversion
def mirror_x(state):
    return f"{state[6]}{state[7]}{state[8]}{state[3]}{state[4]}{state[5]}{state[0]}{state[1]}{state[2]}"
@type_conversion
def all_transformation(state):
    return [rotate_270(state),rotate_180(state),rotate_90(state),mirror_diag_2(state),mirror_diag_1(state),mirror_y(state),mirror_x(state)]


def choice_at_random(state,symbol):
    """
    Given a state and symbol, plays at random on the grid and returns the resulting state
    """
    move = random.sample([i for i in range(len(state)) if state[i]=="-"],1)
    if len(move)==0:
        return state
    return replace_in_str(state,move[0],symbol)

class Opponent:
    """
    BOT Player than places his symbol on the grid based on his strategy function
    """
    def __init__(self,strategy):

        self.strategy = strategy

    def play(self,state,symbol):
        return self.strategy(state,symbol)

def replace_in_str(s,index,symbol):
    temp = list(s)
    temp[index] = symbol
    return "".join(temp)

class Agent:
    """
    Autonomous Agent
    """
    def __init__(self,symbol="X"):

        self.epsilon = .1;
        self.alpha = .1
        self.symbol = symbol
        self.state_table,self.state_table_ref = self.init_state_table(symbol)

    def game_vs_opponent(self,opponent:Opponent,symbol = "X",print_final_grid:bool =False):
        state = "---------"
        not_symbol = "O" if symbol=="X" else "X"
        game_ended = False
        turn = symbol == "X"
        while not game_ended:
            
            if turn:
                #Agent plays
                empty_spaces = [i for i in range(len(state)) if state[i]=="-"]
                possible_state= [] # contains [(nex_state,proba_this_state_is_winning) ... (...) ] will choose the argmax of proba winning
                if len(empty_spaces)==0:
                    game_ended = True
                    break
                for space in empty_spaces:
                    temp = replace_in_str(state,space,symbol) # plays the legal move
                    temp_transformation = all_transformation(temp) #get all the transformation of the new state
                    proba_winning = 0
                    for transfo in temp_transformation:
                        proba_winning = max(proba_winning,self.state_table.get(transfo,0))
                    possible_state.append((temp,proba_winning))
                previous_state = self.state_table_ref.get(state[:]) # store the last state (current) for training (unique key)
                state = sorted(possible_state,key = lambda k : k[1],reverse=True)[0][0] #get the argmax of proba_winning (by sorting along the axis 1 reversed)
                state_unique_key = self.state_table_ref.get(state)
                
                self.state_table[previous_state] = self.state_table[previous_state] + self.alpha * (self.state_table[state_unique_key] - self.state_table[previous_state])
            else:
                state = opponent.play(state=state,symbol=not_symbol)
            turn = not turn
            game_ended = state.replace("-","&").replace(symbol,"&") in winning_state_list.get(not_symbol) or state.replace("-","&").replace(not_symbol,"&") in winning_state_list.get(symbol) or state.count("-")==0
        if print_final_grid:
            pretty_print_grid(state)
  
    def init_state_table(self,symbol="X"):
        """
        Inits the state matrix of the agent
            - Generate all states (1.9e4 possibilty)
            - Only add grid that are valid, .i.e : 
                - No more than 5 Xs and 4 0s (no more Os than Xs)
                - Only one state per set of winning-independant transformation
            - Save the state in a hashmap with the probability of winning (1 if already won, 0 if lost, 0.5 else)
        Also returns a state X state hashmap that points from one state to its unique key in the state_table dict
        """
        state_dict = dict()
        state_table_ref = dict()
        not_symbol = "O" if symbol=="X" else "X"
        cell_state = ["X","O","-"]
        all_line_states = []
        all_grid_states = []
        grid_added = set()
        for c1 in cell_state:
            for c2 in cell_state:
                for c3 in cell_state:
                    all_line_states.append("".join([c1,c2,c3]))
        for l1 in all_line_states:
            for l2 in all_line_states:
                for l3 in all_line_states:
                    all_grid_states.append("".join([l1,l2,l3]))
        
        for state in all_grid_states:
            if not state.count("O")>4 and not state.count("X")>5 and not state.count("O")>state.count("X"):
                # if this is a valid grid
                proba_winning = winning_state.get(symbol).get(state.replace("-","&").replace(not_symbol,"&"))
                proba_losing = winning_state.get(not_symbol).get(state.replace("-","&").replace(symbol,"&"))
                if not (proba_winning and proba_losing):
                    # if the grid is not a winning state for both "X" and "O"
                    winning = 1 if proba_winning else 0 if proba_losing else 0.5
                    rot = all_transformation(state)
                    if state not in grid_added:
                        state_dict[state] = winning
                        state_table_ref[state] = state
                        for r in rot:
                            state_table_ref[r] = state
                    for r in rot:
                        grid_added.add(r)

        return state_dict,state_table_ref

# print([k for k,v in Agent().init_state_table().items() if v==1])

# grid = [["X","X","O"],["X","O","O"],["O","X","X"]]
# pretty_print_grid(grid)
# grid_str = encode_state(grid)

# pretty_print_grid(rotate_270(grid))

a = Agent()
o = Opponent(strategy=choice_at_random)
a.game_vs_opponent(o,print_final_grid=False)
a.game_vs_opponent(o,print_final_grid=False)

# N_game = 1000
# for _ in range(N_game):
#     a.game_vs_opponent(o,print_final_grid=False)

print([(k,v) for k,v in a.state_table.items() if v>0.5 and v<1])