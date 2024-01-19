# DimensionMixerBlock

import torch
from torch import nn, Tensor

#from zeta.nn import (
#    img_to_text,
#    video_to_text,
#    audio_to_text,
#    Attention,
#    FeedForward,
#)
#from local_attention import LocalAttention

class DimensionMixerBlock(nn.Module):
    def __init__(
        self,
        # dim: int,
        # heads: int = 8,
        # dim_head: int = 64,
        # dropout: float = 0.1,
        # window_size: int = 512,
        # causal: bool = True,
        # look_backward: int = 1,
        # look_forward: int = 0,
        # seqlen: int = 1028,
        # ff_mult: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
            self.dim = dim
            self.heads = heads
            self.dim_head = dim_head
            self.dropout = nn.Dropout(dropout)
            self.window_size = window_size
            self.seqlen = seqlen
            self.ff_mult = ff_mult

            inner_dim = dim_head * heads

            self.local_attn = LocalAttention(
                window_size,
                causal,
                look_backward,
                look_forward,
                dropout,
                dim=dim,
                autopad=True,
                *args,
                **kwargs,
            )
        
        self.attn = Attention(
            dim,
            dim_head,
            heads,
            causal,
            flash=False,
            dropout=dropout,
            qk_norm=True,   
        )
        
        self.ffn = FeedForward(
            dim,
            dim,
            ff_mult,
            post_act_ln=True
        )

class ButterflyMLP(nn.Module):
        def __init__(
        self,
        # dim: int,
        # heads: int = 8,
        # dim_head: int = 64,
        # dropout: float = 0.1,
        # window_size: int = 512,
        # causal: bool = True,
        # look_backward: int = 1,
        # look_forward: int = 0,
        # seqlen: int = 1028,
        # ff_mult: int = 4,
        *args,
        **kwargs,
    ):

class ButterflyAttention(nn.Module):
        def __init__(
        self,
        # dim: int,
        # heads: int = 8,
        # dim_head: int = 64,
        # dropout: float = 0.1,
        # window_size: int = 512,
        # causal: bool = True,
        # look_backward: int = 1,
        # look_forward: int = 0,
        # seqlen: int = 1028,
        # ff_mult: int = 4,
        *args,
        **kwargs,
    ):


class BlockLinear(nn.Module):
    def __init__(self, 
        num_blocks: int,
        input_block_dim: int,
        output_block_dim: int,
        self.weight = torch.randn(
                num_blocks: int,
                input_block_dim: int,
                output_block_dim: int
        )
        self.bias = torch.randn(
            num_blocks: int,
            input_block_dim: int,
            output_block_dim: int
        )
    )

    def forward(self, x: Tensor) -> Tensor:
        ## x -> [num_blocks, batch_size, input_block_dim]
        return  torch.batch_matmul(x, self.weight) + self.bias

class BlockMLP(nn.Module):
    def __init__(self,
        input_dim: int,
        layer_block_dims=[]: list,
        actf=nn.GELU
        ):
        self.block_dim = layer_dims[0]
        num_blocks = input+_dim // layer_block_dims[0]

        # Create a block MLP
        self.mlp = nn.Sequential([])
        for i in range(len(layer_block_dims)-1):
            self.mlp += [ BlockLinear ( num_blocks, layer_block_dims[i], layer_block_dims[i+1] ),
                            actf() ]
            self.mlp += self.mlp[:-1]

    def forward(self, x: Tensor) -> Tensor:
        bs, input_dim = x.shape
        x = x.view(bs, -1, self.block_dim).transpose(0,1)
        x = self.mlp(x)
        x = x.transpose(0,1).view(bs,-1)
        return x

# x : Input with shape [batch_size, input_dim]
# block_dims -> size of block in  block_sparse MLP. Usually a factor of input_dim
# block_layers : Layer of block mixing function.
# fn_block : Block mixing function; is parallel non-linear mixer operating per block.
# y : Output with same shape as Input x for simplicty.

block_layers = []
for _ in range(log_base(input_dim, base=block_dim)): # using hidden expansion of 2
    block_layers += [ BlockMLP(input_dim,  [block_dim, block_dim+2, block_dim]) ]

# Using Butterfly Permutation

for i, fn_block in enumerate(block_layers):
    stride = block_size**i if (block_size**(i+1) <= input_dim) else input_dim // block_size
    x = x.view(-1, block_dim, stride).transpose(2,1).view(batch_size, -1)
    x = fn_block(x)
    x = x.view(-1, stride, block_dim).transpose(2, 1).view(batch_size, -1)


return x

# Algorithm 2: Permutation of Butterfly Attention for Transformers

# x : Input with shape [BATCH SIZE (B), SEQUENCE LENGTH (S), MODEL DIMENSION (D)]
# mask : Attention mask (binary) - EITHER: token-wise mask
#  - OR: same size as Attention [-1, num_blocks, num_heads, block_size, block_size]
# block_size (a) : Radix or Block size of Butterfly Attention
# i : Index of layer in butterfly (i.e. 0, 1, ... log_a(S)-1 ; S is Sequence Length)
# transformer : a transformer layer with Attntion and MLP layers (Vaswani et al., 2017)

def permutation():
    B, S, D = x.shape
    for i, transformer in enumerate(transformers):
        stride = block_size**i if (block_size**(i+1) <= S) else S // block_size
        x = x.view(B, -1, block_size, D).transpose(2, 3).view(-1, block_size, D)
        x = transformer(x, mask)
        x = x.view(B, -1, stride, D).transpose(2, 3).view(B, S, D)
    return x