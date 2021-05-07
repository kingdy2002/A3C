import torch
import torch.nn.functional as F
from torch import optim
import NeuralNet
import Advantage
from torch.distributions import Categorical

class Brain(object):
    def __init__(self,GlobalActorNet, GlobalCriticNet,device, socket ,num_action, max_episodes, max_step, batch_size, GAMMA):
        self.GlobalActorNet = GlobalActorNet
        self.GlobalCriticNet = GlobalCriticNet
        self.ActorOptimizer = optim.Adam(self.GlobalActorNet.parameters())
        self.CriticOptimizer = optim.Adam(self.GlobalCriticNet.parameters())
        self.device = device

        self.max_episodes = max_episodes
        self.max_step = max_step


        self.socket = socket
        self.batch_size = batch_size
        self.num_state = 4
        self.GAMMA = GAMMA
        self.kind_action = num_action


    def train(self, GlobalActorNet, GlobalCriticNet):
        train(GlobalActorNet, GlobalCriticNet,num_state = self.num_state, num_action = self.kind_action, device = self.device,
              socket = self.socket,max_episodes =  self.max_episodes,max_step=self.max_step, batch_size=self.batch_size, GAMMA = self.GAMMA )

    def test(self,GlobalActorNet, GlobalCriticNet):
        test(GlobalActorNet,GlobalCriticNet, self.device, self.socket, self.max_episodes, self.max_step)


def train(global_actor, global_critic, num_state, num_action, device, socket, max_episodes, max_step, batch_size, GAMMA):
    local_actor  = NeuralNet.ActorNet(inputs = num_state, outputs = num_action,num_hidden_layers = 3 , hidden_dim = 32).to(device)
    local_critic = NeuralNet.CriticNet(inputs = num_state, outputs = 1,num_hidden_layers = 3 , hidden_dim = 32).to(device)
    """
    print("local actor")
    for param_tensor in local_actor.state_dict():
        print(param_tensor, "\t", local_actor.state_dict()[param_tensor].size())
    print("global actor")
    for param_tensor in global_actor.state_dict():
        print(param_tensor, "\t", global_actor.state_dict()[param_tensor].size())
    """
    local_actor.load_state_dict(global_actor.state_dict())
    local_critic.load_state_dict(global_critic.state_dict())

    entropy_ceof = 0.001

    actor_optimizer = optim.Adam(local_actor.parameters())
    critic_optimizer = optim.Adam(local_critic.parameters())
    memory= Advantage.AdvantageMemory( batch_size, num_state,device=device,GAMMA=GAMMA, kind_action = 1)

    for epi in range(max_episodes):
        local_actor.load_state_dict(global_actor.state_dict())
        local_critic.load_state_dict(global_critic.state_dict())
        state,reward,done,height = socket.getdata()
        local_actor.eval()
        local_critic.eval()

        for step in range(max_step):
            local_actor.eval()
            local_critic.eval()
            #local_actor.train()
            #local_critic.train()
            action_prob = local_actor(torch.from_numpy(state).float().to(device))
            #print(action_prob)
            action_distrib = Categorical(action_prob)
            action = action_distrib.sample()

            #print("start send data")
            socket.senddata(float(action.item()))
            next_state, reward, done, height = socket.getdata()

            if done is True :
                reward = -10
                mask = 0
            else :
                reward = 0
                #reward = (height - 3)/10
                mask = 1

            #print(epi," ",step," ",done)

            #print(action_prob[action],state, next_state, reward , mask)
            memory.input_data(state,next_state,reward,mask)
            state = next_state

            if memory.fill_batch() :

                action_prob = local_actor(memory.states).float().to(device)
                action_distrib = Categorical(action_prob)
                action = action_distrib.sample()
                action = action.unsqueeze(1)
                #print('action is ', action)
                #print('action_prob before is ', action_prob)
                #print('action_prob is ', action_prob)
                action_prob = action_prob.gather(1,action)
                #print('action_prob is ', action_prob)

                state_value = local_critic(memory.states)
                next_state_value = local_critic(memory.next_states)

                #print('memory rewards',memory.rewards)
                #print('next_state value is ',next_state_value)
                Q = memory.rewards + GAMMA * next_state_value.detach()*memory.masks
                #print('memory.rewards',memory.rewards,'GAMMA',GAMMA,'next_state_value.detach()',next_state_value.detach())
                A = Q - state_value
                #print('Q is ',Q,  ' state value is ',state_value,' A is ',A)
                #print(memory.masks, Q)
                #print('next_state_value is ',next_state_value,'Q is ',Q)
                #print('A is ',A)

                local_actor.train()
                local_critic.train()

                critic_optimizer.zero_grad()
                critic_loss = F.smooth_l1_loss(state_value, Q)
                #print('critic loss is ', critic_loss)
                critic_loss.backward()
                critic_optimizer.step()
                global_critic.load_state_dict(local_critic.state_dict())
                log_prob = torch.log(action_prob + 1e-3)
                entropy = -log_prob * action_prob
                #print('action_prob is ', action_prob)
                #print('log_prob is ', log_prob)
                #print('A is ', A)
                #print('entropy_ceof * entropy',entropy_ceof * entropy)
                #print('-A.detach() * log_prob',-A.detach() * log_prob)

                actor_loss = -A.detach() * log_prob - entropy_ceof * entropy
                #print('Q is ', Q, ' state value is ', state_value, ' A is ', A,'actorloss ',actor_loss)
                #print('actor loss is ', -A.detach() * log_prob)
                actor_loss = torch.sum(actor_loss,-1)
                actor_loss = torch.sum(actor_loss, -1)
                #print('actor_loss is ', actor_loss)
                actor_loss /= len(action_prob)
                #print('actor_loss is ', actor_loss)
                #print('actor loss is ',actor_loss)
                actor_loss.backward()
                actor_optimizer.step()
                global_actor.load_state_dict(local_actor.state_dict())

            if done is True :
                print('epi : ', epi, " is end and step is :",step)
                break

def test(global_actor,global_critic, device , socket, episodes,max_step) :

    global_actor.eval()
    for epi in range(episodes) :
        state, reward, done, height = socket.getdata()
        for step in range(max_step) :
            v = global_critic(torch.from_numpy(state).float().to(device))
            print('value is ',v, ' reward is ',(height - 3)/10)
            action_prob = global_actor(torch.from_numpy(state).float().to(device))
            action_distrib = Categorical(action_prob)
            action = action_distrib.sample()

            socket.senddata(float(action.item()))
            state, reward, done, height = socket.getdata()

            if done is True :
                break



