import numpy as np
import torch
import torch.functional as F
import torch.nn as nn


class Act_op(nn.Module):  # Custom activation function
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        # return x ** 50  # or F.relu(x) * F.relu(1-x)
        return (F.relu(x)) ** 3


class DNNNetwork(nn.Module):  # DNN 0: ReLU; 1: Tanh; 2:Sin; 3:x**50; 4:Sigmoid
    def __init__(self, R, w_Univ0, b_Univ0):
        super(DNNNetwork, self).__init__()
        self.block3 = nn.Sequential()
        self.block = nn.Sequential()
        self.asi = True if R["asi"] else False

        for i in range(len(R["full_net"]) - 2):
            d_linear = nn.Linear(R["full_net"][i], R["full_net"][i + 1])
            d_linear.weight.data = torch.nn.Parameter(torch.tensor(w_Univ0[i]))
            d_linear.bias.data = torch.nn.Parameter(torch.tensor(b_Univ0[i]))
            self.block3.add_module("linear" + str(i), d_linear)

            self.block.add_module("linear" + str(i), d_linear)
            if R["ActFuc"] == 0:
                self.block.add_module("relu" + str(i), nn.ReLU())
                self.block3.add_module("relu" + str(i), nn.ReLU())
            elif R["ActFuc"] == 1:
                self.block.add_module("tanh" + str(i), nn.Tanh())
                self.block3.add_module("tanh" + str(i), nn.Tanh())
            elif R["ActFuc"] == 3:
                self.block.add_module("relu3" + str(i), Act_op())
                self.block3.add_module("relu3" + str(i), Act_op())
        
        i = len(R["full_net"]) - 2
        d_linear = nn.Linear(R["full_net"][i], R["full_net"][i + 1], bias=False)
        d_linear.weight.data = torch.nn.Parameter(torch.tensor(w_Univ0[i]))
        d_linear.bias.data = torch.nn.Parameter(torch.tensor(b_Univ0[i]))
        self.block.add_module("linear" + str(i), d_linear)
    
        if R["asi"]:
            self.block2 = nn.Sequential()
            for i in range(len(R["full_net"]) - 2):
                d_linear = nn.Linear(R["full_net"][i], R["full_net"][i + 1])
                d_linear.weight.data = torch.nn.Parameter(torch.tensor(w_Univ0[i]))
                d_linear.bias.data = torch.nn.Parameter(torch.tensor(b_Univ0[i]))
            
                # d_linear.weight.data = torch.tensor(w_Univ0[i])
                # d_linear.bias.data = torch.tensor(b_Univ0[i])
                self.block2.add_module("linear2" + str(i), d_linear)
                if R["ActFuc"] == 0:
                    self.block2.add_module("relu2" + str(i), nn.ReLU())
                elif R["ActFuc"] == 1:
                    self.block2.add_module("tanh2" + str(i), nn.Tanh())
                elif R["ActFuc"] == 2:
                    self.block2.add_module("sin2" + str(i), nn.sin())
                elif R["ActFuc"] == 3:
                    self.block2.add_module("**502" + str(i), Act_op())
                elif R["ActFuc"] == 4:
                    self.block2.add_module("sigmoid2" + str(i), nn.sigmoid())
            
            i = len(R["full_net"]) - 2
            d_linear = nn.Linear(R["full_net"][i], R["full_net"][i + 1], bias=False)
            d_linear.weight.data = torch.nn.Parameter(torch.tensor(-w_Univ0[i]))
            d_linear.bias.data = torch.nn.Parameter(torch.tensor(-b_Univ0[i]))
            self.block2.add_module("linear2" + str(i), d_linear)

    def forward(self, x):
        if self.asi:
            out = self.block(x) + self.block2(x)
        else:
            out = self.block(x)
        return out

    def hidden(self, x):
        out = self.block3(x)
        return out
