import matplotlib.pyplot as plt
import numpy as np
from util_func import *
from reward import *
from bandit import *


def qf():
    """
    Function to answer question f) of the coursework
    Testing the basic properties of a 10 arms bandit problem
    """
    n_arms = 10 #number of arms
   
    means = np.random.normal(0.5,0.25,n_arms) #we draw their mean reward from a random distribution
    
    #We can choose between a normal or bernoulli reward distribution

    # brandit_reward_func,fig_name = np.random.normal,"normale"
    brandit_reward_func,fig_name = bernoulli,"bernoulli",

    reward_function = [Reward(brandit_reward_func,m,0.2) for m in means] # Reward will be drawn from a normal distribution with variable mean and sigma 0.2
    
    #The first test is a regular greedy algorithm (epsilon = 0.) with an optimistic value of Q0= 5
    greedy = bandit_problem(n_arms,means= means.tolist(),Q0 = 5.,reward_function = reward_function,epsilon=0.0)
    greedy.run(10000)

    #The second test is a e-greedy algorithm with epsilon set to 0.1 and 0.01
    e_Greedy_0_01= bandit_problem(n_arms,means= means.tolist(),reward_function = reward_function,epsilon=0.01,c=2)
    e_Greedy_0_01.run(10000)

    e_Greedy_0_1= bandit_problem(n_arms,means= means.tolist(),reward_function = reward_function,epsilon=0.1,c=2)
    e_Greedy_0_1.run(10000)


    #We plot the results of the algorithms
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
    """
    Function to answer question g) of the coursework
    We challenge the non-stationarity properties by introducing normal noise to the rewards
    """
    n_arms = 10 #number of arms
   
    means = np.random.normal(0.5,0.25,n_arms) #we draw their mean reward from a random distribution
    
    # brandit_reward_func,fig_name = np.random.normal,"normale"
    brandit_reward_func,fig_name = bernoulli,"bernoulli",

    reward_function = [Reward(brandit_reward_func,m,0.2) for m in means] # Reward will be drawn from a normal distribution with variable mean and sigma 0.2
    
    #First system is an epsilong-greedy algorithm with sample average method
    e_greedy_sample_average = bandit_problem(n_arms,means= means.tolist(),Q0 = 0.,reward_function = reward_function,stationary = False,epsilon=0.1)
    e_greedy_sample_average.run(10000)
    
    #Second system is an epsilon-greedy algorithm with a fixed step-parameter (alpha = 0.1)
    e_greedy_step_param = bandit_problem(n_arms,means= means.tolist(),Q0 = 0.,reward_function = reward_function,epsilon=0.1,stationary=False,step_parameter=10)
    e_greedy_step_param.run(10000)

    #We plot the results
    fig,axs = plt.subplots(1,2,figsize =(8,8),sharey=True)

    e_greedy_sample_average.plot_accuracy(axs[0])
    axs[0].set_title("Sample average method")
    
    e_greedy_step_param.plot_accuracy(axs[1])
    axs[1].set_title(r"Step parameter : 1/$\alpha$=10")
    plt.show()

def get_user_input(values:list,prompt:str,t=str):
    res = "@!#/\n*"
    while res not in values:
        res = input(prompt)
        if res == "" and None in values:
            #Allow none answmer
            return None
        if type!=str:
            try:
                res = t(res)
            except:
                pass
    return res

def bandit_user_input_script():
    print("Welcome the N-bandit problem\n")
    n_arms = get_user_input(list(range(1,101)),"How many arms (positive integer) : ",t=int)

    means = np.random.normal(0.5,0.25,n_arms)
    
    stationary = get_user_input(["Y","N"],"Stationary setting (Y/N) : ")
    stationary = stationary == "Y"
    if stationary:
        distribution = get_user_input(["B","N"],"Reward distribution (B/N) : ")
        brandit_reward_func = bernoulli if distribution=="B" else np.random.normal
        reward_function = [Reward(brandit_reward_func,m,0.2) for m in means]
    else:
        reward_function = [Reward(bernoulli,m,0.2) for m in means] #if non-stationary, the reward functions don't impact the result

    epsilon = get_user_input([x/10 for x in range(10)],"Epsilon factor (0./0.1/../0.9) : ",float)
    Q0 = get_user_input(list(range(10)),"Optimistic starting value (0/1/../10) : ",int)
    c = get_user_input([x * 0.5 for x in range(3,21)] + [None],"UCB parameters c (None/1.5/2.0/2.5/.../10) : ",float)
    step = get_user_input(list(range(10,110,10)) + [None],"Inverse step-parameters alpha (None/10/20/../100) : ",float)
    bandit = bandit_problem(n_arms=n_arms,means=means,reward_function=reward_function,Q0=Q0,epsilon=epsilon,c=c,step_parameter=step,stationary=stationary)
    bandit.run(10000)
    fig,ax = plt.subplots(1,1,figsize =(8,8))
    ax = bandit.plot_accuracy(ax)  
    ax.set_title(r"{}-arms Bandit problem : {}stationnary , $\epsilon=${} , $\alpha=${},c={},$Q_0=${}".format(n_arms,'' if stationary else 'non-',epsilon,step if step else 'sample-average',c if c else "no UCB",Q0))      
    plt.show()

if __name__=="__main__":
    
    bandit_user_input_script()
