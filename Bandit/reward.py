from numpy.random import normal

class Reward:
    """
    Reward object to draw reward from
    Main argument is the draw function to draw the rewards from (can be normal or bernoulli for instance)
    The draw function will have a arguments : mean,sigma/arg , **arg to be versatile
    If the process is non stationnary, the reward are draw from a random walk that advances each time the bandit is selected
    We store the value of the walk in the random_walk_value variable
    """
    def __init__(self,draw_function,mean,*extra_param):

        self.draw_function = draw_function
        self.mean = mean
        self.param = extra_param

        self.random_walk_value = 0

    def draw(self):
        return self.draw_function(self.mean,*self.param)

    def random_walk_increment(self):
        self.random_walk_value += normal(0,0.01)

    def get_random_valk_value(self):
        return self.random_walk_value

    def draw_non_stationnary(self):
        return self.random_walk_value