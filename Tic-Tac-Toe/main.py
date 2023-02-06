import numpy as np
import matplotlib.pyplot as plt
from agent import *
from opponent import *
import os

def clear_csl():
    os.system('cls')

def evalute_perf():
    a = Agent(symbol="X")
    o = Opponent(strategy=choice_attack_defense)

    sample_test = 1e4
    win_rate,lose_rate = a.test_performance(sample_test,o)

    print(f"Win rate : {round(win_rate/sample_test*100,2)}% | Lose rate : {round(lose_rate/sample_test*100,2)}% | Tie rate : {round((1-win_rate/sample_test - lose_rate/sample_test)*100,2)}%")
    
    a.train(opponent=o,N_games=int(1e5))

    win_rate,lose_rate = a.test_performance(sample_test,o)
    print(f"Win rate : {round(win_rate/sample_test*100,2)}% | Lose rate : {round(lose_rate/sample_test*100,2)}% | Tie rate : {round((1-win_rate/sample_test - lose_rate/sample_test)*100,2)}%")
    
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_accuracy_evolution(ax,symbol,opponent,N = 10000,n_points = 25):
    a = Agent(symbol= symbol)
    step_train = int(N/n_points)
    win_rate,lose_rate = [],[]
    avg_pts = 15
    x = []
    for i in range(1,n_points+1):
        w,l = a.test_performance(100,opponent)
        e = (0.1*(1-step_train*i/N))
        win_rate.append(w)
        lose_rate.append(l)
        x.append(i*step_train)
        a.train(opponent,step_train,epsilon_const=e,disable_tqdm = True)

    ax.plot(x,win_rate,color="green",linestyle="dashdot",linewidth="0.25")
    ax.plot(x,lose_rate,color="red",linestyle="dashdot",linewidth=".25")

    avg_win_rate = moving_average(win_rate,avg_pts)
    x_avg= x[:-avg_pts+1]
    ax.plot(x_avg,avg_win_rate,color="green",label="Win-rate")
    
    avg_lose_rate = moving_average(lose_rate,avg_pts)
    ax.plot(x_avg,avg_lose_rate,color="red",label="Lose-rate")
            
    ax.legend()

    return ax

def test_1():
    """
    Plot the evolution of the accuracy in two sample cases
    """
    fig,axs = plt.subplots(2,1,figsize=(16,8),sharex=True)

    axs[0] = plot_accuracy_evolution(axs[0],"X",Opponent(choice_attack_defense),N=2500,n_points=200)
    axs[1] = plot_accuracy_evolution(axs[1],"O",Opponent(choice_attack_defense),N=2500,n_points=200)

    axs[0].set_title("Playing the 'X' vs Type III opponent")
    axs[1].set_title("Playing the 'O' vs Type III opponent")

    plt.show()

def test_2():
    """
    Train an agent one one type of opponent and evaluate its performance against another
    """
    n_test = 1000
    n_train = 10000
    opponents = [Opponent(choice_at_random),Opponent(choice_immediate_best),Opponent(choice_attack_defense)]
    for symbol in ["X","O"]:
        for i_train,o_train in enumerate(opponents):
            for i_test,o_test in enumerate(opponents):
                a = Agent(symbol)
                a.train(o_train,n_train,disable_tqdm=True)
                w,l = a.test_performance(n_test,o_test)
                t = n_test-w-l
                print(f"Playing {symbol} training on Type I{'I'*i_train} testing on Type I{'I'*i_test} (W|L|T): {round(w/n_test*100,2)}%|{round(l/n_test*100,2)}%|{round(t/n_test*100,2)}%")
            print("\n")

def play_against_agent(agent : Agent = None):
    
    if agent is None:
        t = ""
        opponents = [Opponent(choice_at_random),Opponent(choice_immediate_best),Opponent(choice_attack_defense)]
        while t not in ["I","II","III"]:
            t = input("Train the agent against oponent of type I/II/III : ")
        symbol = ""
        while symbol not in ["X","O"]:
            symbol = input("Play as X/O : ")
        not_symbol = "O" if symbol=="X" else "X"
        o = opponents[["I","II","III"].index(t)]
        agent = Agent(not_symbol)
        agent.train(opponent=o,N_games = 10000)
    else:
        not_symbol = agent.symbol
        symbol = "O" if not_symbol=="X" else "X"
    state = "---------"
    count = 0
    turn = symbol=="X"
    while count<9 and not has_won(state,symbol) and not has_won(state,not_symbol):
        clear_csl()
        pretty_print_grid(state,numbers=True)
        if turn:
            move = input("Case number to play (1/2/.../9) : ")
            if move in ["1","2","3","4","5","6","7","8","9"]:
                if state[int(move)-1]=="-":
                    state = replace_in_str(state,int(move)-1,symbol)
                    turn = not turn
        else:
            state = agent.choose_best_move(state,not_symbol)
            turn = not turn
        count +=1
    clear_csl()
    pretty_print_grid(state,numbers=True)
    if has_won(symbol=symbol,state=state):
        print("You have won !")
    elif has_won(symbol=not_symbol,state=state):
        print("Agent has won")
    else:
        print("Tie !")

    return agent


if __name__=="__main__":
    agent = None
    new_game = "Y"
    while new_game=="Y":
        new_game = ""
        agent = play_against_agent(agent=agent)
        while new_game not in ["Y","N"]:
            new_game = input("Play again (Y/N) : ")
        if new_game == "N":
            break
        discard_agent = ""
        while discard_agent not in ["Y","N"]:
            discard_agent = input("Discard Agent (Y/N) : ")
        if discard_agent=="Y":
            agent = None