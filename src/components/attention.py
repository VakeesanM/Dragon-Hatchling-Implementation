import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, n, d, theta=0.01):
        super().__init__()
        self.D_x = nn.Linear(in_features = d, out_features=n)
        self.D_y = nn.Linear(in_features = d, out_features=n)
        self.E = nn.Linear(in_features = n, out_features=d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ReLU = nn.ReLU()
        theta = torch.tensor(float(theta))
        self.c = torch.cos(theta)
        self.s = torch.sin(theta)

    def apply_rotation(self, state, theta):
        
        even = state[..., :, 0::2]
        odd  = state[..., :, 1::2]

        rotated_even = even * self.c + odd * self.s
        rotated_odd  = -even * self.s + odd * self.c

        rotated = torch.empty_like(state)
        rotated[..., 0::2] = rotated_even
        rotated[..., 1::2] = rotated_odd


        return rotated


    def forward(self, prev_val, prev_neuron_activity, prev_state):
        neuron_activity =  prev_neuron_activity + self.ReLU(self.D_x(prev_val))

        attention_state = prev_state + (prev_val[:, :, None] * neuron_activity[:, None,  :])
        attention_state = self.apply_rotation(attention_state)

        reterived_val = torch.einsum("bdn,bn->bd", prev_state, neuron_activity)

        nueron_gate = self.ReLU(self.D_y(self.ln1(reterived_val)))
        output_neurons = nueron_gate * neuron_activity

        val = self.ln2(self.E(output_neurons))

        return val, neuron_activity, attention_state