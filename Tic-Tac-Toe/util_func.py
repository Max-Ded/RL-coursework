import random

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

def replace_in_str(s,index,symbol):
    temp = list(s)
    temp[index] = symbol
    return "".join(temp)

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
