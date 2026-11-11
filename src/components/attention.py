import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, n, d, theta=0.01):
        super().__init__()
        self.D_x = nn.Linear(in_features = n, out_features=d)
        self.D_y = nn.Linear(in_features = n, out_features=d)
        self.E = nn.Linear(in_features = d, out_features=n)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(n)
        self.ReLU = nn.ReLU()

    def apply_rotation(state, theta):
        
        even = state[:, 0::2]
        odd  = state[:, 1::2]

        c = torch.cos(theta)
        s = torch.sin(theta)

        rotated_even = even * c + odd * s
        rotated_odd  = -even * s + odd * c

        rotated = torch.empty_like(state)
        rotated[:, 0::2] = rotated_even
        rotated[:, 1::2] = rotated_odd

        return rotated


    def foward(self, prev_val, prev_neuron_activity, prev_state):
        neuron_activity =  prev_neuron_activity + self.ReLU(self.D_X(prev_val))

        attention_state = prev_state + (prev_val[:, None] @ neuron_activity[None, :])

        attention_state = self.apply_rotation(attention_state, self.theta)

        reterived_val = prev_state @ neuron_activity

        nueron_gate = self.ReLU(self.ln1(reterived_val))

        output_neurons = nueron_gate * neuron_activity

        val = self.ln2(self.E @ output_neurons)

        return val, neuron_activity, attention_state




if __name__ == '__main__':
    model = Attention(8192, 256)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
