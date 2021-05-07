import Agent
import  Advantage
import NeuralNet
import ClientSocket
import numpy as np
import time
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import StepLR,ExponentialLR
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
import math
import sys
from torch import optim
from PIL import Image
import urllib
import glob
import random
from collections import namedtuple
from sklearn import preprocessing
import torch.multiprocessing as mp
from torchsummary import summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    num_action = 2
    num_state = 4
    num_process = 5

    global_Actor = NeuralNet.ActorNet(inputs = num_state, outputs = num_action,num_hidden_layers = 2 , hidden_dim = 8).to(device)
    #summary(global_Actor, input_size=(10,num_state))
    global_Critic = NeuralNet.CriticNet(inputs = num_state, outputs = 1,num_hidden_layers = 2 , hidden_dim = 8).to(device)
    #summary(global_Critic, input_size=(10,num_state))
    batch_size = 64
    GAMMA = 0.95
    max_episodes = 5000
    max_step = 1000
    global_Actor.share_memory()
    global_Critic.share_memory()

    processes = []
    processes_socket =[]
    processes_agent = []
    mp.set_start_method('spawn')
    print("MP start method:",mp.get_start_method())

    ip = '110.76.78.109'
    port = 1111
    for rank in range(num_process):
        processes_socket.append(0)
        processes_socket[rank] = ClientSocket.MySocket(port,  'f',  'ffff?f')
        processes_agent.append(0)
        processes_agent[rank]= Agent.Brain(GlobalActorNet = global_Actor, GlobalCriticNet = global_Critic,device =device , socket= processes_socket[rank] ,num_action = num_action, max_episodes = max_episodes,
                    max_step= max_step, batch_size=batch_size, GAMMA=GAMMA)
        p = mp.Process(target=processes_agent[rank].train, args=())
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()