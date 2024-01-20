"""Tests for main.py"""

import torch
import torch.nn as nn
from .dimension_mixer import main
from .dimension_mixer.main import BlockLinear, BlockMLP

def test_block_linear():
    block_linear = BlockLinear(2, 3, 4)
    x = torch.randn(2, 5, 3)
    output = block_linear(x)
    assert output.shape == (2, 5, 4)

def test_block_mlp():
    block_mlp = BlockMLP(6, [2, 3, 4])
    x = torch.randn(5, 6)
    output = block_mlp(x)
    assert output.shape == (5, 6)
    
def test_permutation():
    x = torch.randn(10, 20, 30)  # batch size 10, sequence length 20, model dimension 30
    transformers = [nn.TransformerEncoderLayer(d_model=30, nhead=5) for _ in range(3)]
    block_size = 5
    output = permutation(x, transformers, block_size)
    assert output.shape == x.shape
