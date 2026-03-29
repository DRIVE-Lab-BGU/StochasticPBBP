class noise():
    def __init__(self, action_dim, max_action, horizon):
        self.action_dim = action_dim
        self.max_action = max_action
        self.horizon = horizon
    def get_smaller_1(self , start_noise , end_noise , step   ):
        noise = step * (end_noise - start_noise) / self.horizon + start_noise
        return noise

    def get_smaller_2(self , start_noise , end_noise , step   ):    
        noise  = max( (step / self.horizon)*start_noise  , end_noise)
        return noise
    @staticmethod
    def constant_noise( noise = 0 ):
        return noise
    
    def get_noise(self):
        return 0