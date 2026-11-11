import torch
import torch.nn as nn
from components.attention import Attention


class BDH(nn.Module):

    def __init__(self, n, d, heads, vocab_size, theta):
        super().__init__()
        self.n = n
        self.d = d
        self.heads = heads
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)
        self.ln = nn.LayerNorm(d)
        self.attn = Attention(n, d, theta)
        self.fc = nn.Sequential(
            nn.Linear(d, d//2),
            nn.Dropout(0.3),
            nn.LayerNorm(d//2, vocab_size)
        )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, X):
        B, T = X.shape
        val = self.ln(self.embedding(X))
        neuron_activity = torch.zeros(self.n)
        attention_states = [ torch.zeros(B, self.d, self.n, device=X.device) for _ in range(self.heads)]

        outputs = []
        for t in range(T):
            val = self.ln(self.embedding(X[:, t]))                
            neuron_activity = torch.zeros(B, self.n, device=X.device) 

            for l in range(self.heads):
                val, neuron_activity, attention_states[l] = self.attn(
                    val, neuron_activity, attention_states[l]
                )

            outputs.append(self.fc(val))                          

        probs = self.softmax(torch.stack(outputs, dim=1))
        return torch.argmax(probs)                         