import numpy as np
import random

class bandit_problem:

    def __init__(self,n_arms:int,means:list,reward_function:list,Q0=0,epsilon:float=0,c=None,step_parameter=None,stationary = True):
        """
        Environnement set-up with n_arms bandit arm :
            Their disitrbution is :  N(mu_i,sigma) where means = [mu_1,...,...mu_n_arms]
                                  :  Bernoulli (mean) where means = [mu_1,...,...mu_n_arms]
        
        We have multiple parameters : 
                n_arms : number of arms in the problem
                mean : list of mean of the rewards distributions
                reward_function : reward obj list to draw the rewards from
                Q0 : optimistic initial value for the q function (initialize all the values to Q0)
                epsilon : e-greedy algorithm, choose exploration over exploitation with e probability
                c : Parameters for the UCB computation (no UCB if c is None)
                step_parameter : step parameter for the learning process, if None will be set to standard 1/(n+1)
                stationnary : if True we draw the reward according to the standard process, else we set the rewards as random walk with normal 0,0.01 increments
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
        # retrieve the reward obj associated with the bandit and draw a reward according to a random walk
        reward = self.reward_function[index].get_random_valk_value() # Get the reward from this bandit arm
        self.reward_function[index].random_walk_increment() # add noise to the reward
        self.best_reward_list.append(max([self.reward_function[i].get_random_valk_value() for i in range(self.n_arms)])) # add the reward drawn for future plotting
        return reward
        
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
       
        reward = self.draw_reward(choice) if self.stationary else self.draw_reward_with_noise(choice) #If the process is stationnary we process the reward using the reward obj, else we use the random walk

        alpha = self.step_parameter if self.step_parameter else  (self.n[choice]+1) # Use the step parameter if given or else 1/(N_t(a)+1)

        self.q[choice] = self.q[choice] + 1/alpha * (reward-self.q[choice])
        self.n[choice]+=1
        self.rewards = np.append(self.rewards,reward)

    def run(self,N):
        #Simulate N rounds to learn from them
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
