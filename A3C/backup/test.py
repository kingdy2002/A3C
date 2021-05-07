import NeuralNet
from torchsummary import summary
import torch
import multiprocessing as mp
def compute():
    for j in range(5) :
        print(j)
lis = []


for i in range(5) :
    p = mp.Process(target=compute, args=())
    lis.append(p)
    p.start()
for p in lis:
    p.join()