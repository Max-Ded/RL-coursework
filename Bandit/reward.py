from numpy.random import normal

class reward:

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