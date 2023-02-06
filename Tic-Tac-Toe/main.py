import numpy as np
import matplotlib.pyplot as plt
from agent import *
from opponent import *

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

if __name__=="__main__":

    fig,axs = plt.subplots(2,1,figsize=(16,8),sharex=True)

    axs[0] = plot_accuracy_evolution(axs[0],"X",Opponent(choice_attack_defense),N=2500,n_points=200)
    axs[1] = plot_accuracy_evolution(axs[1],"O",Opponent(choice_attack_defense),N=2500,n_points=200)

    axs[0].set_title("Playing the 'X' vs Type III opponent")
    axs[1].set_title("Playing the 'O' vs Type III opponent")

    plt.show()