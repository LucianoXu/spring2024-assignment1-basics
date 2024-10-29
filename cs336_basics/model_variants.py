from .model import *

class TransformerBlock_ParallelLayers(TransformerBlock):
    def forward(self, x: torch.Tensor):

        temp_x = self.rmsnorm1(x)
        x1 = F.dropout(self.multi_atten(temp_x), self.residual_pdrop)

        temp_x = self.rmsnorm2(x)
        x2 = F.dropout(self.ffn(temp_x), self.residual_pdrop)

        return x + x1 + x2

class TransformerLM_ParallelLayers(torch.nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 dim: int,
                 num_heads: int,
                 d_ff: int,
                 attn_pdrop: float|None = None, 
                 residual_pdrop: float|None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_lenght = context_length
        self.num_layers = num_layers

        self.dim = dim
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop if residual_pdrop is not None else 0.


        self.embedding = nn.Embedding(vocab_size, dim)
        self.abs_pos_embedding = nn.Parameter(torch.rand(context_length, dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerBlock_ParallelLayers(
                dim,
                num_heads,
                d_ff,
                attn_pdrop,
                residual_pdrop
            ))

        self.rmsnorm = RMSNorm(dim)

        self.linear = nn.Linear(dim, vocab_size, bias=False)

    
    def forward(self, x: torch.Tensor):

        token_embedding = self.embedding(x)
        seq_len = x.shape[-1]
        pos_embedding = self.abs_pos_embedding[:seq_len, :]

        x = F.dropout(token_embedding + pos_embedding, self.residual_pdrop)

        for layer in self.layers:
            x = layer(x)

        x = self.linear(self.rmsnorm(x))

        return x
