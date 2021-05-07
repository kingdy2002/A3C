import torch.multiprocessing as mp
import Agent
import  Advantage
import NeuralNet
import ClientSocket
import numpy as np
import time
import pandas as pd
import os
import torch

def main():

    processes = []
    processes_socket = []
    processes_agent = []

    device = torch.device('cpu')
    num_action = 2
    num_state = 4
    num_process = 1

    batch_size = 64
    GAMMA = 0.95
    max_episodes = 5000
    max_step = 1000

    global_Actor = NeuralNet.ActorNet(inputs=num_state, outputs=num_action, num_hidden_layers=2, hidden_dim=8).to(device)
    global_Critic = NeuralNet.CriticNet(inputs=num_state, outputs=1, num_hidden_layers=2, hidden_dim=8).to(device)

    dic = torch.load(f"D:/modelDict/actor/modelDict.pt")
    global_Actor.load_state_dict(torch.load("D:/modelDict/actor/modelDict.pt"))
    global_Critic.load_state_dict(torch.load("D:/modelDict/critic/modelDict.pt"))

    port = 1111
    for rank in range(num_process):
        processes_socket.append(0)
        processes_socket[rank] = ClientSocket.MySocket(port,  'f',  'ffff?f')
        processes_agent.append(0)
        processes_agent[rank ]= Agent.Brain(GlobalActorNet = global_Actor, GlobalCriticNet = global_Critic ,device =device , socket= processes_socket[rank] ,num_action=num_action, max_episodes=max_episodes,
                                            max_step=max_step, batch_size=batch_size, GAMMA=GAMMA)
        p = mp.Process(target=processes_agent[rank].test, args=(global_Actor,global_Critic))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()