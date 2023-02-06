import matplotlib.pyplot as plt
import numpy as np
from util_func import *
from reward import *
from bandit import *


def qf():
    n_arms = 10 #number of arms
   
    means = np.random.normal(0.5,0.25,n_arms) #we draw their mean reward from a random distribution
    
    # brandit_reward_func,fig_name = np.random.normal,"normale"
    brandit_reward_func,fig_name = bernoulli,"bernoulli",

    reward_function = [reward(brandit_reward_func,m,0.2) for m in means] # Reward will be drawn from a normal distribution with variable mean and sigma 0.2
    
    #The first test is a regular greedy algorithm (epsilon = 0.) with an optimistic value of Q0= 5
    greedy = bandit_problem(n_arms,means= means.tolist(),Q0 = 5.,reward_function = reward_function,epsilon=0.0)
    greedy.run(10000)

    #The second test is a e-greedy algorithm with epsilon set to 0.1 and 0.01
    e_Greedy_0_01= bandit_problem(n_arms,means= means.tolist(),reward_function = reward_function,epsilon=0.01,c=2)
    e_Greedy_0_01.run(10000)

    e_Greedy_0_1= bandit_problem(n_arms,means= means.tolist(),reward_function = reward_function,epsilon=0.1,c=2)
    e_Greedy_0_1.run(10000)

    fig,axs = plt.subplots(3,1,figsize =(8,8) ,sharex = True)

    greedy.plot_accuracy(axs[0])
    axs[0].set_title("Greedy algorithm - $Q_0 = 5$")
    axs[0].set_ylabel("Accuracy")
    
    e_Greedy_0_01.plot_accuracy(axs[1])
    axs[1].set_title("$\epsilon$-greedy - $\epsilon=0.01 , c=2$")
    axs[1].set_ylabel("Accuracy")

    e_Greedy_0_1.plot_accuracy(axs[2])
    axs[2].set_title("$\epsilon$-greedy - $\epsilon=0.1 , c=2$")
    axs[2].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(f"image/{fig_name}.png")
    plt.show()

def qg():
    n_arms = 10 #number of arms
   
    means = np.random.normal(0.5,0.25,n_arms) #we draw their mean reward from a random distribution
    
    # brandit_reward_func,fig_name = np.random.normal,"normale"
    brandit_reward_func,fig_name = bernoulli,"bernoulli",

    reward_function = [reward(brandit_reward_func,m,0.2) for m in means] # Reward will be drawn from a normal distribution with variable mean and sigma 0.2
    
    #The first test is a regular greedy algorithm (epsilon = 0.) with an optimistic value of Q0= 5
    e_greedy_sample_average = bandit_problem(n_arms,means= means.tolist(),Q0 = 0.,reward_function = reward_function,stationary = False,epsilon=0.1)
    e_greedy_sample_average.run(10000)
    
    e_greedy_step_param = bandit_problem(n_arms,means= means.tolist(),Q0 = 0.,reward_function = reward_function,epsilon=0.1,stationary=False,step_parameter=10)
    e_greedy_step_param.run(10000)
    fig,axs = plt.subplots(1,2,figsize =(8,8),sharey=True)

    e_greedy_sample_average.plot_accuracy(axs[0])
    axs[0].set_title("Sample average method")
    
    e_greedy_step_param.plot_accuracy(axs[1])
    axs[1].set_title(r"Step parameter : 1/$\alpha$=10")
    plt.show()

if __name__=="__main__":

    qf()
