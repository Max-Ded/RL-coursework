import numpy as np
import random
from tqdm import tqdm
from agent import *
from opponent import *

def evalute_perf():
    a = Agent(symbol="O")
    o = Opponent(strategy=choice_at_random)

    sample_test = 1e4
    win_rate,lose_rate = a.test_performance(sample_test,o)

    print(f"Win rate : {round(win_rate/sample_test*100,2)}% | Lose rate : {round(lose_rate/sample_test*100,2)}% | Tie rate : {round((1-win_rate/sample_test - lose_rate/sample_test)*100,2)}%")
    
    a.train(opponent=o,N_games=int(1e5))

    win_rate,lose_rate = a.test_performance(sample_test,o)
    print(f"Win rate : {round(win_rate/sample_test*100,2)}% | Lose rate : {round(lose_rate/sample_test*100,2)}% | Tie rate : {round((1-win_rate/sample_test - lose_rate/sample_test)*100,2)}%")
    

if __name__=="__main__":

    evalute_perf()