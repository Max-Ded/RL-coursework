import numpy as np
import random
from tqdm import tqdm

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

def has_won(state,symbol):
    for i in range(3):
        if state[i]==symbol and state[3+i]==symbol and state[6+i]==symbol:
            return True
        if state[3*i:3*(i+1)]==symbol*3:
            return True
    if state[0]==symbol and state[4]==symbol and state[8]==symbol:
        return True
    if state[2]==symbol and state[4]==symbol and state[6]==symbol:
        return True
    return False


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
    # a_{13}a_{23}a_{33}a_{12}a_{22}a_{32}a_{11}a_{21}a_{31}
    return f"{state[2]}{state[5]}{state[8]}{state[1]}{state[4]}{state[7]}{state[0]}{state[3]}{state[6]}"
@type_conversion
def rotate_180(state):
    # a_{33}a_{32}a_{31}a_{23}a_{22}a_{21}a_{13}a_{12}a_{11}
    return f"{state[8]}{state[7]}{state[6]}{state[5]}{state[4]}{state[3]}{state[2]}{state[1]}{state[0]}"
@type_conversion
def rotate_90(state):
    # a_{31}a_{21}a_{11}a_{22}a_{32}a_{12}a_{33}a_{23}a_{13}
    return f"{state[6]}{state[3]}{state[0]}{state[4]}{state[7]}{state[1]}{state[8]}{state[5]}{state[2]}"
@type_conversion
def mirror_diag_2(state):
    # a_{33}a_{23}a_{13}a_{32}a_{22}a_{12}a_{31}a_{21}a_{11}
    return f"{state[8]}{state[5]}{state[2]}{state[7]}{state[4]}{state[1]}{state[6]}{state[3]}{state[0]}"
@type_conversion
def mirror_diag_1(state):
    # a_{11}a_{21}a_{31}a_{12}a_{22}a_{32}a_{31}a_{32}a_{33}
    return f"{state[0]}{state[3]}{state[6]}{state[1]}{state[4]}{state[7]}{state[2]}{state[5]}{state[8]}"
@type_conversion
def mirror_y(state):
    # a_{13}a_{12}a_{11}a_{23}a_{22}a_{21}a_{33}a_{32}a_{31}
    return f"{state[2]}{state[1]}{state[0]}{state[5]}{state[4]}{state[3]}{state[8]}{state[7]}{state[6]}"
@type_conversion
def mirror_x(state):
    #a_{31}a_{32}a_{33}a_{21}a_{22}a_{23}a_{11}a_{12}a_{13}
    return f"{state[6]}{state[7]}{state[8]}{state[3]}{state[4]}{state[5]}{state[0]}{state[1]}{state[2]}"
@type_conversion
def all_transformation(state):
    return [rotate_270(state),rotate_180(state),rotate_90(state),mirror_diag_2(state),mirror_diag_1(state),mirror_y(state),mirror_x(state)]


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

def replace_in_str(s,index,symbol):
    temp = list(s)
    temp[index] = symbol
    return "".join(temp)

class Agent:
    """
    Autonomous Agent
    """
    def __init__(self,symbol="X"):
        self.alpha = 2
        self.symbol = symbol
        self.state_table,self.state_table_ref = self.init_state_table(symbol)

    def test_performance(self,N_games,opponent:Opponent):
        """
        Make the agent play against an opponent with learning and returns the win/lose rate
        """
        sample_test = int(N_games)
        win_rate =0
        lose_rate = 0
        for _ in range(sample_test):
            game = self.game_vs_opponent(opponent,print_final_grid=False,learn=False)
            if game=="X":
                win_rate+=1
            elif game=="O":
                lose_rate+=1
        return win_rate,lose_rate    

    def train(self,opponent:Opponent,N_games:int,epsilon=0.1,symbol = "X"):
        for step in tqdm(range(N_games+1)):
            self.game_vs_opponent(opponent=opponent,symbol=symbol,epsilon = (epsilon*(1-step/N_games)))

    def choose_best_move(self,state,symbol="X",print_step=False):
        empty_spaces = [i for i in range(len(state)) if state[i]=="-"]
        random.shuffle(empty_spaces)
        possible_state= [] # contains [(nex_state,proba_this_state_is_winning) ... (...) ] will choose the argmax of proba winning
        for space in empty_spaces:
            temp = replace_in_str(state,space,symbol) # plays the legal move
            unique_temp = self.state_table_ref.get(temp)
            possible_state.append((temp,self.state_table.get(unique_temp)))
        state = sorted(possible_state,key = lambda k : k[1],reverse=True)[0][0] #get the argmax of proba_winning (by sorting along the axis 1 reversed)
        if print_step:
            print("Chosing between : ")
            for g,s in possible_state:
                pretty_print_grid(g)
                print(s)
                print("__")
            print("___________")
        return state

    def game_vs_opponent(self,opponent:Opponent,symbol = "X",epsilon=0.1,print_final_grid:bool =False,learn:bool=True,print_game_history=False):
        state = "---------"
        not_symbol = "O" if symbol=="X" else "X"
        game_ended = False
        turn = symbol == "X"
        game_history = []
        previous_state= "---------"
        while not game_ended:
            if turn:
                #Agent plays                
                previous_state = self.state_table_ref.get(state[:],state[:]) # store the last state (current) for training (unique key)
                
                if random.random()>epsilon:
                    #With proba 1-e => we select the best possible move
                    state = self.choose_best_move(state,symbol=self.symbol,print_step=print_game_history)
                else:
                    #with proba e , we select a move at random to explore
                    state = choice_attack_defense(state,self.symbol)  
            else:
                state = opponent.play(state=state,symbol=not_symbol)

            if learn:
                #we update the table with lookback parameter alpha
                state_unique_key = self.state_table_ref.get(state)
                self.state_table[previous_state] = self.state_table[previous_state] + self.alpha * (self.state_table[state_unique_key] - self.state_table[previous_state])
            game_history.append(state)
            turn = not turn
            game_ended = has_won(state,symbol=symbol) or has_won(state,symbol=not_symbol) or state.count("-")==0
        if print_final_grid:
            pretty_print_grid(state)
        if print_game_history:
            for s in game_history:
                pretty_print_grid(s)
        symbol_winner =  has_won(state,symbol=symbol)
        not_symbol_winner =  has_won(state,symbol=not_symbol)
        if not symbol_winner and not not_symbol_winner:
            #if nobody has won
            return "-"
        elif symbol_winner:
            return symbol
        else:
            return not_symbol
  
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
            if not state.count("O")>4 and not state.count("X")>5 and state.count("X") - state.count("O")<=2 and state.count("X") - state.count("O")>=0:
                # if this is a valid grid
                proba_winning = has_won(state,symbol=symbol)
                proba_losing = has_won(state,symbol=not_symbol)
                if not (proba_winning and proba_losing):
                    # if the grid is not a winning state for both "X" and "O"
                    winning = 1 if proba_winning else 0 if proba_losing else 0.5 #Initial value of the state
                    rot = all_transformation(state)
                    if state not in grid_added:
                        #if the state or any variation is never seen before
                        state_dict[state] = winning #we add it in the state_table with its value
                        state_table_ref[state] = state #we point to itself in the ref table
                        grid_added.add(state)
                        for r in rot:
                            #all the rotation will point to this state in the ref table
                            state_table_ref[r] = state
                        for r in rot:
                            #we will have seen all the rotation
                            grid_added.add(r)

        return state_dict,state_table_ref



if __name__=="__main__":

    a = Agent(symbol="X")
    o = Opponent(strategy=choice_attack_defense)
    # random.seed(1)
    # a.game_vs_opponent(o,print_final_grid=False,learn=True)
    # random.seed(1)
    # a.game_vs_opponent(o,print_final_grid=False,learn=True)
    # random.seed(1)
    # a.game_vs_opponent(o,print_final_grid=False,learn=True)
    # random.seed(1)
    # a.game_vs_opponent(o,print_final_grid=False,learn=True)
    # random.seed(1)
    # a.game_vs_opponent(o,print_final_grid=False,learn=True)
    # random.seed(1)
    # a.game_vs_opponent(o,print_final_grid=False,learn=True)
    # random.seed(1)
    # print("new game")
    # a.game_vs_opponent(o,print_final_grid=False,learn=True)
    sample_test = 1e4
    win_rate,lose_rate = a.test_performance(sample_test,o)

    print(f"Win rate : {round(win_rate/sample_test*100,2)}% | Lose rate : {round(lose_rate/sample_test*100,2)}%")
    
    a.train(opponent=o,N_games=int(1e5))

    win_rate,lose_rate = a.test_performance(sample_test,o)
    print(f"Win rate : {round(win_rate/sample_test*100,2)}% | Lose rate : {round(lose_rate/sample_test*100,2)}%")
    
    print(a.game_vs_opponent(o,print_game_history=True,learn=True))

    print([(k,v) for k,v in a.state_table.items() if v>0.5 and v<1])