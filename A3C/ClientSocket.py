import socket
from struct import *
import torch
import numpy as np


class MySocket :
    def __init__(self,port, send_pack,reciv_pack) :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ClientSocket = socket.socket(socket.AF_INET , socket.SOCK_STREAM)
        self.ClientSocket.connect(('110.76.78.109',port))
        self.ClientSocket.setblocking(True)
        self.send_pack = send_pack #EX) 'ff'
        self.reciv_pack = reciv_pack  # EX) 'ff'

    def senddata(self, action):
        data = pack(self.send_pack, action)
        #print("I send data")
        self.ClientSocket.send(data)

    def getdata(self):
        #print("product getdata")
        data = self.ClientSocket.recv(1024)
        #print(data)
        pktFormat = self.reciv_pack
        pktSize = calcsize(pktFormat)
        #print("pktSize is ",pktSize)
        data1, data2, data3, data4,  done, hight = unpack(pktFormat, data[:pktSize])
        #print(data1,data2,data3,data4)
        #train_data = dqn.DataNormalization(np.array([data1, data2, data3, data4]))
        train_data = np.array([data1,data2, data3, data4])
        #train_data = torch.unsqueeze(train_data, 0)
        return (train_data, 0, done, hight)


