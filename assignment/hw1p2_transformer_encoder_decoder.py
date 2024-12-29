# -*- coding: utf-8 -*-
"""hw1p2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12UsAWsJRrTv4tqF3X6GkTlfkfsOxQYRK

<h1 align="center" style="color:green;font-size: 3em;">Homework 1 Part 2:
Implementing Transformers From Scratch Using PyTorch</h1>


# Introduction

In this homework, you will implement the encoder, decoder, and full transformer as described in the "Attention is All You Need" paper from scratch using PyTorch.

**Instructions:**
- Follow the notebook sections to implement various components of the transformer.
- Code cells marked with `TODO` are parts that you need to complete.
- Ensure your code runs correctly by the end of the notebook.
- Please make sure your nobtebook is named **hw1p2** when you submit and please submit both ".py" and ".ipynb" version to Gradescope


*Below we have provided the correct code from hw1p1. This is because you will need it to build you encoder and decoder, and we want to make sure you are not penalized for past work.*
"""

# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
warnings.simplefilter("ignore")
print(torch.__version__)



# Set the seed value
seed_value = 0

# For CPU
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# For GPU (if using CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        """
        Args:
            max_seq_len: maximum length of input sequence
            embed_model_dim: dimension of embeddings
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        # TODO: Initialize the positional encoding matrix using the above equation
        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(self.embed_dim // 2):
                pe[pos, 2*i] = math.sin(pos / (10000 ** (2*i / self.embed_dim)))
                pe[pos, 2*i+1] = math.cos(pos / (10000 ** (2*i / self.embed_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output vector with positional encoding added
        """

        # Weight the embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)

        # TODO: Add positional encoding to the input embeddings
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len, :], requires_grad=False)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embedding vector output
            n_heads: number of self-attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    # 512 dim
        self.n_heads = n_heads   # 8 heads
        self.single_head_dim = embed_dim // n_heads   # 512 / 8 = 64, each key, query, and value head will be 64d

        # EDIT: These were all changed to be of shape embed_dim x embed_dim instead of single_head_dim x single_head_dim
        # TODO: Initialize key, query, and value matrices
        self.query_matrix = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.key_matrix = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value_matrix = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out = nn.Linear(n_heads * self.single_head_dim, embed_dim)

    def forward(self, key, query, value, mask=None):
        """
        Args:
            key: key vector
            query: query vector
            value: value vector
            mask: mask for decoder

        Returns:
            output: vector from multi-head attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)

        # EDIT: this was moved from below reshape to above.
        # TODO: Apply linear transformations
        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        # TODO: Reshape key, query, and value
        k = k.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        q = q.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        v = v.view(batch_size, seq_length, self.n_heads, self.single_head_dim)


        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # TODO: Compute attention scores
        k_adjusted = k.transpose(-1, -2)
        product = torch.matmul(q, k_adjusted)

        if mask is not None:
            mask = mask.unsqueeze(1)  # Adds an extra dimension for the heads
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / math.sqrt(self.single_head_dim)
        scores = F.softmax(product, dim=-1)

        # TODO: Compute weighted sum of value vectors and run it through the last layer
        scores = torch.matmul(scores, v)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query, self.single_head_dim * self.n_heads)
        output = self.out(concat)

        return output

"""# Part 4: Encoder
<img src="https://www.researchgate.net/profile/Ehsan-Amjadian/publication/352239001/figure/fig1/AS:1033334390013952@1623377525434/Detailed-view-of-a-transformer-encoder-block-It-first-passes-the-input-through-an.jpg" width="250" height="500">

In this section, we will fully implement the encoder.

## 4.1 Encoder Class

<h2>Steps for the Encoder:</h2>

**Step 1:**

   The input (padded tokens of the sentence) is passed through the embedding layer and positional encoding layer.
```
Code Hint:
If the input is of size 32x10 (batch size=32 and sequence length=10), after passing through the embedding layer, it becomes 32x10x512.
This output is added to the corresponding positional encoding vector, producing a 32x10x512 output that is passed to the multi-head attention.
```

**Step 2:**
  
  The processed input is passed through the multi-head attention layer to create a useful representational matrix.
```
Code Hint:
The input to the multi-head attention is 32x10x512. Key, query, and value vectors are generated, ultimately producing a 32x10x512 output.
```
**Step 3:**
  
  The output from the multi-head attention is added to its input and then normalized.
```
Code Hint:
The output from the multi-head attention (32x10x512) is added to the input (32x10x512) and then normalized.
```
**Step 4:**
  
  The normalized output passes through a feed-forward layer and another normalization layer with a residual connection from the input of the feed-forward layer.
```
Code Hint:
The normalized output (32x10x512) is passed through two linear layers: 32x10x512 -> 32x10x2048 -> 32x10x512.
Finally, a residual connection is added, and the layer is normalized.
This produces a 32x10x512 dimensional vector as the encoder's output.
```

<hr>
<h3>Task:</h3>

Implement the `TransformerBlock` and `TransformerEncoder` classes. Complete the `__init__` and `forward` methods for the full encoder functionality.
"""

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: factor determining output dimension of the linear layer
           n_heads: number of attention heads
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector

        Returns:
           norm2_out: output of transformer block
        """
        # TODO: Calculate attention output using self.attention
        attention_out = self.attention.forward(key, query, value)

        # TODO: Add residual connection and normalize
        attention_residual_out = self.dropout1(query+attention_out)
        norm1_out = self.norm1(attention_residual_out)

        # TODO: Pass through feed forward layer
        feed_fwd_out = self.feed_forward(norm1_out)

        # TODO: Add residual connection and normalize
        feed_fwd_residual_out = self.dropout2(norm1_out+ feed_fwd_out)
        norm2_out = self.norm2(feed_fwd_residual_out)

        return norm2_out


class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor determining the number of linear layers in feed-forward layer
        n_heads: number of heads in multi-head attention

    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)])

    def forward(self, x):
        # TODO: Apply the embedding layer
        embed_out = self.embedding_layer(x)

        # TODO: Apply positional encoding
        out = self.positional_encoder.forward(embed_out)

        # TODO: Pass through each TransformerBlock
        for layer in self.layers:
            out = layer.forward(out, out, out)

        # Return the final output
        return out  # 32x10x512

def test_transformer_block():
    embed_dim = 512
    n_heads = 8
    expansion_factor = 4

    # Define input shapes: batch_size x seq_length x embed_dim
    batch_size = 32
    seq_length = 10

    # Create random input tensors
    key = torch.rand(batch_size, seq_length, embed_dim)
    query = torch.rand(batch_size, seq_length, embed_dim)
    value = torch.rand(batch_size, seq_length, embed_dim)

    # Create the TransformerBlock
    transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    # Pass the inputs through the transformer block
    output = transformer_block(key, query, value)

    # Check the output shape: should be [batch_size, seq_length, embed_dim]
    assert output.shape == (batch_size, seq_length, embed_dim), \
        f"Expected output shape {(batch_size, seq_length, embed_dim)}, but got {output.shape}"

    print("TransformerBlock test passed!")

test_transformer_block()

def test_transformer_encoder():
    seq_len = 10
    vocab_size = 10000
    embed_dim = 512
    num_layers = 2
    n_heads = 8
    expansion_factor = 4

    # Define input shape: batch_size x seq_length
    batch_size = 32

    # Create random input tensor (sequence of token indices)
    input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Create the TransformerEncoder
    transformer_encoder = TransformerEncoder(seq_len, vocab_size, embed_dim, num_layers, expansion_factor, n_heads)

    # Pass the input through the encoder
    output = transformer_encoder(input_seq)

    # Check the output shape: should be [batch_size, seq_len, embed_dim]
    assert output.shape == (batch_size, seq_len, embed_dim), \
        f"Expected output shape {(batch_size, seq_len, embed_dim)}, but got {output.shape}"

    print("TransformerEncoder test passed!")

test_transformer_encoder()

"""# Part 5: Decoder

<img src="https://discuss.pytorch.org/uploads/default/optimized/3X/8/e/8e5d039948b8970e6b25395cb207febc82ba320a_2_177x500.png" width = "200" height="600">

Now we have gone through most parts of the encoder. Let us dive into the components of the decoder. We will use the output of the encoder to generate key and value vectors for the decoder. There are two kinds of multi-head attention in the decoder: decoder self-attention and encoder-decoder attention. Don't worry, we will go step by step.

## 5.1: Decoder Class

<h2>Steps for the Decoder:</h2>

**Step 1:**

The decoder first passes the target sequence through the embedding and positional encoding layers to create an embedding vector of dimension 1x512 for each word in the target sequence.

```
Code Hint:
Suppose we have a sequence length of 10, a batch size of 32, and an embedding vector dimension of 512. The input of size 32x10 to the embedding matrix produces an output of dimension 32x10x512. This gets added to the positional encoding of the same dimension, resulting in a 32x10x512 output.
```
**Step 2:**

The embedding output is then passed through a multi-head attention layer, which creates key, query, and value matrices from the target input and produces an output vector. This time, a mask is used with multi-head attention.

**Why mask?**

A mask is used to prevent a word from attending to future words in the target sequence. For example, in the sentence "I am a student," we do not want the word "a" to attend to the word "student."
```
Code Hint:
To create the attention mask, we use a triangular matrix with 1s and 0s. For example, a triangular matrix for a sequence length of 5 looks like this:

1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1

After the key is multiplied by the query, you should fill all zero positions with a very small number (e.g., -1e20) to avoid division errors.
```
**Step 3:**

As before, we have an add and norm layer where the output of the embedding is added to the attention output and then normalized.

**Step 4:**

Next, we have another multi-head attention layer followed by an add and norm layer. This multi-head attention is called encoder-decoder attention. For this attention, key and value vectors are created from the encoder output, while the query is created from the output of the previous decoder layer.
```
Code Hint:
The encoder output (32x10x512) is used to generate key and value vectors. Similarly, the query matrix is generated from the output of the previous decoder layer (32x10x512).

The output from the previous add and norm layer is added to the encoder-decoder attention output and then normalized.
```
**Step 5:**

Next, we have a feed-forward layer (linear layer) with an add and norm layer similar to the one in the encoder.

**Step 6:**

Finally, we create a linear layer with a size equal to the vocabulary size of the target corpus, followed by a softmax function to get the probability of each word.

<hr>
<h3>Task:</h3>

1. Implement the forward pass for `DecoderBlock`, making sure to apply masked attention with the given mask.
2. Implement the forward pass for `TransformerDecoder`, including embedding, positional encoding, and the final linear layer with softmax.
"""

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: factor determining output dimension of the linear layer
           n_heads: number of attention heads
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi-head attention

        Returns:
           out: output of transformer block
        """
        # TODO: Implement masked attention with the given mask
        attention = self.attention.forward(x, x, x , mask=mask) #the x is from the input of the decoder part
        # TODO: Add residual connection and apply normalization
        value = self.norm(attention+ x)

        # TODO: Pass through the transformer block
        out = self.transformer_block.forward(key, query, value) #key and values are from encoder

        return out

class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
        """
        Args:
           target_vocab_size: vocabulary size of the target
           embed_dim: dimension of embedding
           seq_len: length of input sequence
           num_layers: number of decoder layers
           expansion_factor: factor determining the number of linear layers in the feed-forward layer
           n_heads: number of heads in multi-head attention
        """
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_dim, expansion_factor=4, n_heads=8) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        """
        Args:
            x: input vector from target
            enc_out: output from encoder layer
            trg_mask: mask for decoder self-attention

        Returns:
            out: output vector
        """
        # TODO: Apply the embedding layer
        x = self.word_embedding(x)

        # TODO: Apply positional encoding
        x = self.position_embedding(x)

        # TODO: Apply dropout
        x = self.dropout(x)

        # TODO: Pass through each DecoderBlock with mask
        for layer in self.layers:
            x = layer.forward(enc_out, enc_out, x, mask)

        # TODO: Apply final linear layer and softmax
        out = F.softmax(self.fc_out(x), dim=-1)

        return out

from torch.testing import assert_close

batch_size = 32
seq_len = 10
embed_dim = 512
target_vocab_size = 10000
n_heads = 8
num_layers = 2
expansion_factor = 4

# Test case 1: Full Transformer Decoder functionality
def test_transformer_decoder_full_pass():
    """
    Tests the forward pass of TransformerDecoder with mask and checks if output has expected shape and behavior.
    """
    # Initialize a TransformerDecoder
    decoder = TransformerDecoder(
        target_vocab_size=target_vocab_size,
        embed_dim=embed_dim,
        seq_len=seq_len,
        num_layers=num_layers,
        expansion_factor=expansion_factor,
        n_heads=n_heads
    )

    x = torch.randint(0, target_vocab_size, (batch_size, seq_len))
    enc_out = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, seq_len, seq_len).bool()

    # Forward pass through the transformer decoder
    output = decoder(x, enc_out, mask)

    # Assert the output has the correct shape (batch_size, seq_len, target_vocab_size)
    assert output.shape == (batch_size, seq_len, target_vocab_size), \
        f"Expected {(batch_size, seq_len, target_vocab_size)}, but got {output.shape}"

    # Check that the output is a valid probability distribution (i.e., each row sums to 1 after softmax)
    assert_close(output.sum(dim=-1), torch.ones(batch_size, seq_len), rtol=1e-2, atol=1e-2)

    print("TransformerDecoder passed the full forward pass test!")


test_transformer_decoder_full_pass()

# Test case 2: DecoderBlock functionality with masking
def test_decoder_block_attention_with_mask():
    """
    Tests the attention mechanism in DecoderBlock, ensuring that masking is applied properly and output shape is correct.
    """
    # Initialize a DecoderBlock
    decoder_block = DecoderBlock(embed_dim=embed_dim, expansion_factor=expansion_factor, n_heads=n_heads)

    key = torch.randn(batch_size, seq_len, embed_dim)
    query = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # Create a random mask that simulates some positions being masked (0 for masked positions)
    mask = torch.randint(0, 2, (batch_size, seq_len, seq_len)).bool()

    # Forward pass through the DecoderBlock with mask
    output = decoder_block(key, query, value, mask)

    # The correct shape: (batch_size, seq_len, embed_dim)
    assert output.shape == (batch_size, seq_len, embed_dim), \
        f"Expected {(batch_size, seq_len, embed_dim)}, but got {output.shape}"

    print("DecoderBlock passed the attention mechanism test with masking!")


test_decoder_block_attention_with_mask()

"""That's all for HW1 Part 2! Please submit both the .py and .ipynb on Gradescope and note any score there may not be your final score on the assignment. Final scores will be released when the overall homework is due."""