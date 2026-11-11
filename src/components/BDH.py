from attention import Attention
import torch
import torch.nn as nn



class BDH(nn.Module):

    def __init__(self, n, d, heads, vocab_size, theta):
        super().__init__()
        self.n = n
        self.d = d
        self.heads = heads
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)
        self.ln = nn.LayerNorm(d)
        self.attn = Attention(n, d)
        self.fc = nn.Sequential(
            nn.Linear(d, d//2),
            nn.Dropout(0.3),
            nn.LayerNorm(d//2, vocab_size)
        )
    def forward(self, X):

        val = self.ln(self.embedding(X))
        neuron_activity = torch.zeros(self.n)
        attention_state = torch.zeros(self.d,self.n)

        for _ in range(self.heads):
            val, neuron_activity, attention_state = self.attn(val, neuron_activity, attention_state)

        return self.fc(val)




