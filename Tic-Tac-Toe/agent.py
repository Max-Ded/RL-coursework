from util_func import *
from opponent import Opponent,choice_attack_defense
import numpy as np
import random
from tqdm import tqdm


class Agent:
    """
    Autonomous Agent
    """
    def __init__(self,symbol="X"):
        self.alpha = 0.2
        self.symbol = symbol
        self.state_table,self.state_table_ref = self.init_state_table(symbol)

    def test_performance(self,N_games,opponent:Opponent):
        """
        Make the agent play against an opponent with learning and returns the win/lose rate
        """
        sample_test = int(N_games)
        not_symbol = "O" if self.symbol=="X" else "X"
        win_rate =0
        lose_rate = 0
        for _ in range(sample_test):
            game = self.game_vs_opponent(opponent,print_final_grid=False,learn=False)
            if game==self.symbol:
                win_rate+=1
            elif game==not_symbol:
                lose_rate+=1
        return win_rate,lose_rate    

    def train(self,opponent:Opponent,N_games:int,epsilon=0.1):
        for step in tqdm(range(N_games+1)):
            self.game_vs_opponent(opponent=opponent,epsilon = (epsilon*(1-step/N_games)))

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

    def game_vs_opponent(self,opponent:Opponent,epsilon=0.1,print_final_grid:bool =False,learn:bool=True,print_game_history=False):
        state = "---------"
        symbol = self.symbol
        not_symbol = "O" if symbol=="X" else "X"
        game_ended = False
        turn = symbol == "X"
        game_history = []
        while not game_ended:
            if turn:
                #Agent plays                                
                if random.random()>epsilon:
                    #With proba 1-e => we select the best possible move
                    state = self.choose_best_move(state,symbol=symbol,print_step=print_game_history)
                else:
                    #with proba e , we select a move at random to explore
                    state = choice_attack_defense(state,self.symbol)  
            else:
                state = opponent.play(state=state,symbol=not_symbol)

            game_history.append(state)
            turn = not turn
            game_ended = has_won(state,symbol=symbol) or has_won(state,symbol=not_symbol) or state.count("-")==0
        
        if learn:
            for i in range(len(game_history)-1,0,-1):
                s,s_before = self.state_table_ref.get(game_history[i]),self.state_table_ref.get(game_history[i-1])
                self.state_table[s_before] += self.alpha * (self.state_table.get(s) - self.state_table.get(s_before))
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