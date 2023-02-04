import random
import matplotlib.pyplot as plt
import numpy as np

def bernoulli(mean,*args):
    return 1 if random.random() < mean else 0

class reward:

    def __init__(self,draw_function,mean,*extra_param):

        self.draw_function = draw_function
        self.mean = mean
        self.param = extra_param

        self.random_walk_value = 0

    def draw(self):
        return self.draw_function(self.mean,*self.param)

    def random_walk_increment(self):
        self.random_walk_value += np.random.normal(0,0.01)

    def draw_non_stationnary(self):
        return self.random_walk_value

class bandit_problem:

    def __init__(self,n_arms:int,means:list,reward_function:list,Q0=0,epsilon:float=0,c=None,step_parameter=None,stationary = True):
        """
        Environnement set-up with n_arms bandit arm :
            Their disitrbution is :  N(mu_i,sigma) where means = [mu_1,...,...mu_n_arms]
                                  :  Bernoulli (mean) where means = [mu_1,...,...mu_n_arms]
        """
        self.n_arms = n_arms
        self.means = means

        self.n = np.array([0]*n_arms)
        self.q = np.array([Q0]*n_arms) #Optimistic init at 5.

        self.epsilon = epsilon
        self.rewards = np.array([])
        self.reward_function = reward_function
        self.c = c
        self.step_parameter = step_parameter

        self.stationary = stationary

        if not stationary:
            self.best_reward_list = []

        self.count = 0

    def draw_reward(self,index):
        # retrieve the reward obj associated with the bandit and draw a reward from it with the *draw* method
        reward = self.reward_function[index].draw()
        return reward

    def draw_reward_with_noise(self,index):
        # retrieve the reward obj associated with the bandit and draw a reward from it with the *draw* method
        for rf in self.reward_function:
            rf.random_walk_increment()
        self.best_reward_list.append(max([x.random_walk_value for x in self.reward_function]))
        return self.reward_function[index].random_walk_value
        
    def round(self):
        #We choose the action with best value of Q with proba 1-epsilon, we choose it at random with proba epsilon
        
        UCB = not self.c is None # If we have a parameter c, we use the UCB algorithm
        if random.random()<self.epsilon:
            choice = random.sample(range(self.n_arms),1)[0]
        else:
            # if we use the UCB algorithm, we find the action which maximizes = q_t(a) + c \/ln(t)/n_t(a)
            # else we maximize q_t(a)
            t = len(self.q)
            to_maximize = self.q + self.c * np.sqrt(np.log(t)/(self.n+1)) if UCB else self.q
            best_bandit = [i for i in range(self.n_arms) if to_maximize[i]==max(to_maximize)]
            choice =   random.sample(best_bandit,1)[0]
       
        reward = self.draw_reward(choice) if self.stationary else self.draw_reward_with_noise(choice)

        alpha = self.step_parameter if self.step_parameter else  (self.n[choice]+1) # Use the step parameter if given or else 1/(N_t(a)+1)

        self.q[choice] = self.q[choice] + 1/alpha * (reward-self.q[choice])
        self.n[choice]+=1
        self.rewards = np.append(self.rewards,reward)

    def run(self,N):
        for _ in range(N):
            self.round()

    def plot_accuracy(self,ax,plot_mean = True):
        """
        To run after training
        Accuracy = Σ reward / (N*reward_max)
        """
        N = len(self.rewards)

        choice =   random.sample([i for i in range(self.n_arms) if self.means[i]==max(self.means)],1)[0]
        reward_max = [self.draw_reward(choice) for i in range(N)]

        reward_max_mean = [max(self.means)] * N
        
        rewards_rolling_sum = np.cumsum(self.rewards)
        max_rewards_rolling_sum = np.cumsum(reward_max)
        reward_max_mean_rolling_sum = np.cumsum(reward_max_mean)

        if self.stationary:
            accuracy = rewards_rolling_sum/max_rewards_rolling_sum
            accuracy_mean = rewards_rolling_sum/reward_max_mean_rolling_sum
        else:
            accuracy = rewards_rolling_sum/np.cumsum(self.best_reward_list)

        ax.plot(list(range(N)),accuracy,color = "red",label="Realisation of R*")
        if self.stationary:
            ax.plot(list(range(N)),accuracy_mean,color="blue",linestyle="--",label="Mean of R*")
        ax.legend()
        
        return ax

    def __str__(self):
        return f"Means : {self.means}\nN : {self.n}\nQ : {self.q}"


def rolling_sum_numpy(a:np.array):
    return np.apply_along_axis(lambda index : a[:index].sum(),0,list(range(a.size)))


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
    e_greedy_sample_average.run(100000)
    
    e_greedy_step_param = bandit_problem(n_arms,means= means.tolist(),Q0 = 0.,reward_function = reward_function,epsilon=0.1,stationary=False,step_parameter=10)
    e_greedy_step_param.run(100000)
    fig,axs = plt.subplots(1,2,figsize =(8,8),sharey=True)

    e_greedy_sample_average.plot_accuracy(axs[0])
    axs[0].set_title("Sample average method")
    
    e_greedy_step_param.plot_accuracy(axs[1])
    axs[1].set_title(r"Step parameter : 1/$\alpha$=10")
    plt.show()

if __name__=="__main__":

    qg()
