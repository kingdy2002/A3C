import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module) :
    def __init__(self, inputs, outputs, num_hidden_layers , hidden_dim = None):
        super(Net, self).__init__()

        if hidden_dim == None :
            hidden_dim = inputs * 4

        self.first_layer = nn.Sequential(
            nn.Linear(inputs,hidden_dim),
            nn.Dropout(p = 0.3),
            nn.ReLU()
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Dropout(p = 0.3),
            nn.ReLU()
        )
        self.last_layer = nn.Sequential(
            nn.Linear(hidden_dim,outputs)
        )
        self.layers = nn.ModuleList()
        self.layers.append(self.first_layer)
        for i in range(num_hidden_layers) :
            self.layers.append(self.hidden_layer)
        self.layers.append(self.last_layer)

    def forward(self, x):
        for layer in self.layers :
            x = layer(x)
        return x
class CriticNet(nn.Module) :
    def __init__(self, inputs, outputs,num_hidden_layers , hidden_dim = None ):
        super(CriticNet, self).__init__()

        if hidden_dim == None :
            hidden_dim = inputs * 4

        self.first_layer = nn.Sequential(
            nn.Linear(inputs,hidden_dim),
            nn.Dropout(p = 0.3),
            nn.ReLU()
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Dropout(p = 0.3),
            nn.ReLU()
        )
        self.last_before_layer = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        self.last_layer = nn.Sequential(
            nn.Linear(hidden_dim//2,outputs)
        )
        self.layers = nn.ModuleList()
        self.layers.append(self.first_layer)
        for i in range(num_hidden_layers-1) :
            self.layers.append(self.hidden_layer)
        self.layers.append(self.last_before_layer)
        self.layers.append(self.last_layer)

    def forward(self, x):
        for layer in self.layers :
            x = layer(x)
        return x

class ActorNet(nn.Module) :
    def __init__(self, inputs, outputs,num_hidden_layers , hidden_dim = None):
        super(ActorNet, self).__init__()

        if hidden_dim == None:
            hidden_dim = inputs * 4

        self.first_layer = nn.Sequential(
            nn.Linear(inputs, hidden_dim),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        self.last_before_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        self.last_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, outputs)
        )
        self.layers = nn.ModuleList()
        self.layers.append(self.first_layer)
        for i in range(num_hidden_layers - 1):
            self.layers.append(self.hidden_layer)
        self.layers.append(self.last_before_layer)
        self.layers.append(self.last_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x,dim = -1)
        return x

class Critics_Actor(nn.Module) :
    def __init__(self, inputs, actor_outputs, num_hidden_layers_root, num_hidden_layers_end,critic_outputs = 1):
        super(Critics_Actor, self).__init__()
        self.RootNet = Net(inputs,inputs*4,num_hidden_layers_root,hidden_dim=inputs*4)
        self.Critic = CriticNet(inputs*4,num_hidden_layers_end,outputs=critic_outputs,hidden_dim=2)
        self.Actor = ActorNet(inputs*4,actor_outputs, num_hidden_layers_end,hidden_dim=2)

    def forward(self, x) :
        x = self.RootNet(x)
        actor = self.Actor(x)
        critic = self.Critic(x)

        return actor , critic

def save_model(model, filepath = None):
    if filepath == None:
        filepath = 'D:/modelDict'

    torch.save(model.state_dict(), f"{filepath}/modelDict.pt")

def NpToTensor(numpy) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.from_numpy(numpy).type(torch.LongTensor).to(device)

def TensorToNp(tensor) :
    tensor.to(torch.device("cpu"))
    num = tensor.cpu().numpy()
    return num;
