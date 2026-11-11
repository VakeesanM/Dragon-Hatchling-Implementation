import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, n, d, heads=10, theta=0.01):


        self.D_x = nn.Linear(in_features = n, out_features=d)
        self.D_y = nn.Linear(in_features = n, out_features=d)
        self.E = nn.Linear(in_features = d, out_features=n)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(n)
        self.ReLU = nn.ReLU()
        self.U = self.create_rope_u(n, theta)

    def create_rope_u(self, n, theta):
        U = torch.eye(n)

        for i in range(0, n, 2):
            c = torch.cos(theta)
            s = torch.sin(theta)
            U[i:i+2, i:i+2] = torch.tensor([
                [c, -s],
                [s,  c]
            ])

        return U


    def foward(self, prev_val, prev_neuron_activity, prev_state):
        neuron_activity =  prev_neuron_activity + self.ReLU(self.D_X(prev_val))

        attention_state = prev_state + (prev_val[:, None] @ neuron_activity[None, :]) @ self.U

        reterived_val = prev_state @ neuron_activity

        nueron_gate = self.ReLU(self.ln1(reterived_val))

        output_neurons = nueron_gate * neuron_activity

        val = self.ln2(self.E @ output_neurons)

        return val, neuron_activity, attention_state






