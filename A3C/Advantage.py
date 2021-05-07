import torch


class AdvantageMemory(object):
    def __init__(self, batch_size, num_state,GAMMA,device, kind_action = 1):
        self.device = device
        self.states = torch.zeros(batch_size , num_state).to(self.device)
        self.next_states = torch.zeros(batch_size, num_state).to(self.device)
        self.masks = torch.ones(batch_size , kind_action).to(self.device)
        self.rewards = torch.zeros(batch_size, kind_action).to(self.device)
        self.action_prob = [0]*batch_size

        self.returns = torch.zeros(batch_size +1 , 1)
        self.index = 0
        self.GAMMA = GAMMA
        self.batch_size = batch_size

    def input_data(self, state ,next_state, reward, mask):
        self.states[self.index] = torch.from_numpy(state).float().to(self.device)
        self.next_states[self.index] = torch.from_numpy(next_state).float().to(self.device)
        self.masks[self.index] = torch.tensor([mask]).to(self.device)
        self.rewards[self.index] =  torch.tensor([reward]).to(self.device)
        self.index = (self.index +1) % self.batch_size


    def compute_returns(self, returns):

        self.returns[-1] = returns
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * self.GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]

    def fill_batch(self):
        if self.index == self.batch_size -1 :
            return True
        return False


