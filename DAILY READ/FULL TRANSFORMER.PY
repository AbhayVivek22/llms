"""
TRANSFORMER ARCHITECTURE: COMPLETE FLOW VISUALIZATION

                          INPUT SEQUENCE                      OUTPUT SEQUENCE
                               │                                    │
                               ▼                                    ▼
┌─────────────────────────────────────────────┐  ┌─────────────────────────────────────────────┐
│               TOKEN EMBEDDING               │  │               TOKEN EMBEDDING               │
└───────────────────────┬─────────────────────┘  └───────────────────────┬─────────────────────┘
                        │                                                │
                        ▼                                                ▼
┌─────────────────────────────────────────────┐  ┌─────────────────────────────────────────────┐
│             POSITIONAL ENCODING             │  │             POSITIONAL ENCODING             │
└───────────────────────┬─────────────────────┘  └───────────────────────┬─────────────────────┘
                        │                                                │
                        ▼                                                ▼
┌─────────────────────────────────────────────┐  ┌─────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────┐ │  │ ┌─────────────────────────────────────────┐ │
│ │           MULTI-HEAD ATTENTION          │ │  │ │          MASKED MULTI-HEAD ATTENTION    │ │
│ └──────────────────────┬──────────────────┘ │  │ └──────────────────────┬──────────────────┘ │
│                        │                    │  │                        │                    │
│                        ▼                    │  │                        ▼                    │
│ ┌─────────────────────────────────────────┐ │  │ ┌─────────────────────────────────────────┐ │
│ │          ADD & NORM (RESIDUAL)          │ │  │ │          ADD & NORM (RESIDUAL)          │ │
│ └──────────────────────┬──────────────────┘ │  │ └──────────────────────┬──────────────────┘ │
│                        │                    │  │                        │                    │
│                        ▼                    │  │                        ▼                    │
│ ┌─────────────────────────────────────────┐ │  │ ┌─────────────────────────────────────────┐ │
│ │          FEED-FORWARD NETWORK           │ │  │ │           MULTI-HEAD ATTENTION          │◄─┼────┐
│ └──────────────────────┬──────────────────┘ │  │ │          (ENCODER-DECODER)              │ │    │
│                        │                    │  │ └──────────────────────┬──────────────────┘ │    │
│                        ▼                    │  │                        │                    │    │
│ ┌─────────────────────────────────────────┐ │  │                        ▼                    │    │
│ │          ADD & NORM (RESIDUAL)          │ │  │ ┌─────────────────────────────────────────┐ │    │
│ └──────────────────────┬──────────────────┘ │  │ │          ADD & NORM (RESIDUAL)          │ │    │
│                        │                    │  │ └──────────────────────┬──────────────────┘ │    │
└────────────────────────┼────────────────────┘  │                        │                    │    │
                         │                        │                        ▼                    │    │
                         │                        │ ┌─────────────────────────────────────────┐ │    │
                         │                        │ │          FEED-FORWARD NETWORK           │ │    │
                         │                        │ └──────────────────────┬──────────────────┘ │    │
                         │                        │                        │                    │    │
                         │                        │                        ▼                    │    │
                         │                        │ ┌─────────────────────────────────────────┐ │    │
                         │                        │ │          ADD & NORM (RESIDUAL)          │ │    │
                         │                        │ └──────────────────────┬──────────────────┘ │    │
                         │                        └────────────────────────┼────────────────────┘    │
                         │                                                 │                         │
                         │                                                 ▼                         │
                         │                        ┌─────────────────────────────────────────────┐    │
                         └────────────────────────►                LINEAR LAYER                 │    │
                                                  └───────────────────────┬─────────────────────┘    │
                                                                          │                          │
                                                                          ▼                          │
                                                  ┌─────────────────────────────────────────────┐    │
                                                  │                  SOFTMAX                    │    │
                                                  └───────────────────────┬─────────────────────┘    │
                                                                          │                          │
                                                                          ▼                          │
                                                  ┌─────────────────────────────────────────────┐    │
                                                  │              OUTPUT PROBABILITIES           │    │
                                                  └─────────────────────────────────────────────┘    │
                                                                                                     │
                                                                                                     │
DATA FLOW EXPLANATION:                                                                               │
                                                                                                     │
1. Input tokens are embedded and positional information is added                                     │
2. Encoder processes input through multiple layers of self-attention and feed-forward networks       │
3. Decoder takes both the encoder output and (shifted) target sequence                               │
4. Decoder uses masked self-attention to prevent looking at future tokens                            │
5. Cross-attention connects the encoder and decoder, allowing decoder to focus on relevant input     │
6. Final linear layer and softmax convert decoder output to probabilities over vocabulary            │
7. During training: compare with actual next tokens                                                  │
8. During inference: generate one token at a time, feeding back into decoder                         │
                                                                                                     │
MULTI-HEAD ATTENTION PROCESS:                                                                        │
                                                                                                     │
   Query, Key, Value                                                                                 │
        │                                                                                            │
        ▼                                                                                            │
┌────────────────┐                                                                                   │
│ Linear Layers  │ (Project to multiple heads)                                                       │
└───────┬────────┘                                                                                   │
        │                                                                                            │
        ▼                                                                                            │
┌────────────────┐      ┌───────────────┐                                                            │
│ Scaled Dot     │──────► Apply Mask    │ (Optional)                                                 │
│ Product        │      └───────┬───────┘                                                            │
└───────┬────────┘              │                                                                    │
        │                       │                                                                    │
        ▼                       ▼                                                                    │
┌────────────────┐      ┌───────────────┐                                                            │
│ Softmax        │◄─────┤               │                                                            │
└───────┬────────┘      └───────────────┘                                                            │
        │                                                                                            │
        ▼                                                                                            │
┌────────────────┐                                                                                   │
│ Matrix Multiply│ (with Values)                                                                     │
└───────┬────────┘                                                                                   │
        │                                                                                            │
        ▼                                                                                            │
┌────────────────┐                                                                                   │
│ Concatenate    │ (All heads)                                                                       │
└───────┬────────┘                                                                                   │
        │                                                                                            │
        ▼                                                                                            │
┌────────────────┐                                                                                   │
│ Linear Layer   │ (Final projection)                                                                │
└───────┬────────┘                                                                                   │
        │                                                                                            │
        ▼                                                                                            │
    Attention                                                                                        │
      Output                                                                                         │
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEmbedding(nn.Module):
    """
    Combines token embedding and positional encoding in one layer.
    
    This layer takes integer token IDs and converts them to continuous vector 
    representations, then adds positional information to these vectors so the
    model can distinguish between different positions in the sequence.
    """
    def __init__(self, vocab_size, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        # Token embedding layer converts integer IDs to vectors of size d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix once during initialization
        # This creates a matrix of shape [max_seq_length, d_model] where each
        # row corresponds to a position in the sequence
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create position indices [0, 1, 2, ..., max_seq_length-1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create division terms for the sinusoidal functions
        # These terms control the frequency of the sine/cosine waves
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        # This creates a unique pattern for each position that varies smoothly
        # with position but is different for each dimension
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions get sine
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions get cosine
        
        # Add batch dimension for easier broadcasting later
        pe = pe.unsqueeze(0)  # Shape: [1, max_seq_length, d_model]
        
        # Register as buffer (not a parameter) so it's saved in the state_dict
        # but not updated during backpropagation
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Process input token IDs through embeddings and add positional encoding.
        
        Args:
            x: Tensor of token IDs, shape [batch_size, seq_length]
            
        Returns:
            Tensor of token embeddings with positional encoding,
            shape [batch_size, seq_length, d_model]
        """
        # Convert token IDs to embeddings
        # Shape: [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        x = self.token_embedding(x) * math.sqrt(self.d_model)  # Scale embeddings
        
        # Add positional encoding
        # The PE tensor is sliced to match the sequence length of x
        # Shape after addition: [batch_size, seq_length, d_model]
        x = x + self.pe[:, :x.size(1), :]
        
        # Apply dropout to the sum of embeddings and positional encodings
        # This helps prevent overfitting
        return self.dropout(x)


class Transformer(nn.Module):
    """
    Complete transformer model with encoder and decoder components integrated.
    
    The Transformer processes an input sequence to generate an output sequence,
    using self-attention mechanisms to capture relationships between tokens and
    cross-attention to connect the encoder and decoder.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_seq_length=5000, dropout=0.1):
        """
        Initialize the transformer model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Dimension of model (embedding dimension)
            num_heads: Number of attention heads
            num_layers: Number of encoder and decoder layers
            d_ff: Dimension of feed-forward networks
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout rate
        """
        super().__init__()
        
        # Embedding layers for source and target sequences
        # These convert token IDs to vectors and add positional information
        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_seq_length, dropout)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_seq_length, dropout)
        
        # Stack of encoder layers
        # Each layer processes the output of the previous layer
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # Stack of decoder layers
        # Each layer processes the output of the previous layer and encoder output
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # Final linear layer to project decoder output to vocabulary size
        # This converts the decoder's output vectors to logits over the vocabulary
        self.final_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Layer normalization for encoder and decoder outputs
        # This helps stabilize training
        self.norm = nn.LayerNorm(d_model)
        
    def encode(self, src, src_mask=None):
        """
        Encode the source sequence using the encoder stack.
        
        Args:
            src: Source sequence tensor, shape [batch_size, src_seq_length]
            src_mask: Mask for padding tokens, shape [batch_size, 1, 1, src_seq_length]
            
        Returns:
            Encoder output tensor, shape [batch_size, src_seq_length, d_model]
        """
        # Convert token IDs to embeddings with positional encoding
        # Shape: [batch_size, src_seq_length] -> [batch_size, src_seq_length, d_model]
        x = self.src_embedding(src)
        
        # Pass through each encoder layer sequentially
        # Each layer updates the representation based on self-attention
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
            
        # Apply final layer normalization
        # This ensures the encoder output has stable statistics
        return self.norm(x)
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Decode with encoder output (memory) and target sequence.
        
        Args:
            tgt: Target sequence tensor, shape [batch_size, tgt_seq_length]
            memory: Encoder output tensor, shape [batch_size, src_seq_length, d_model]
            tgt_mask: Mask for target sequence (prevents looking ahead),
                     shape [batch_size, 1, tgt_seq_length, tgt_seq_length]
            memory_mask: Mask for padding in encoder output,
                        shape [batch_size, 1, 1, src_seq_length]
            
        Returns:
            Decoder output tensor, shape [batch_size, tgt_seq_length, d_model]
            Attention weights from all decoder layers
        """
        # Convert target token IDs to embeddings with positional encoding
        # Shape: [batch_size, tgt_seq_length] -> [batch_size, tgt_seq_length, d_model]
        x = self.tgt_embedding(tgt)
        
        # Store attention weights from each layer if needed for visualization
        attentions = []
        
        # Pass through each decoder layer sequentially
        # Each layer updates the representation based on self-attention and
        # cross-attention with the encoder output
        for layer in self.decoder_layers:
            x, attn = layer(x, memory, tgt_mask, memory_mask)
            attentions.append(attn)
            
        # Apply final layer normalization
        # This ensures the decoder output has stable statistics
        return self.norm(x), attentions
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Full forward pass through the transformer.
        
        Args:
            src: Source sequence tensor, shape [batch_size, src_seq_length]
            tgt: Target sequence tensor, shape [batch_size, tgt_seq_length]
            src_mask: Mask for padding tokens in source
            tgt_mask: Mask for target sequence (prevents looking ahead)
            memory_mask: Mask for padding tokens in source (for cross-attention)
            
        Returns:
            Output logits, shape [batch_size, tgt_seq_length, tgt_vocab_size]
            Attention weights from all decoder layers
        """
        # Encode source sequence
        # This creates contextualized representations of the input
        memory = self.encode(src, src_mask)
        
        # Decode target sequence using encoder output
        # This generates contextualized representations of the output tokens
        output, attentions = self.decode(tgt, memory, tgt_mask, memory_mask)
        
        # Project to vocabulary size
        # This converts each token representation to a probability distribution
        # over possible next tokens
        return self.final_out(output), attentions


class EncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feed-forward network.
    
    Each encoder layer allows tokens to attend to all other tokens in the
    input sequence, then processes each token individually through a feed-forward
    network. Residual connections and layer normalization help with training.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Dimension of model (embedding dimension)
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
        """
        super().__init__()
        # Multi-head attention mechanism for self-attention
        # This allows tokens to gather information from other tokens
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise feed-forward network
        # This processes each token position independently
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),        # Expand dimension
            nn.ReLU(),                       # Apply non-linearity
            nn.Dropout(dropout),             # Apply dropout for regularization
            nn.Linear(d_ff, d_model)         # Project back to original dimension
        )
        
        # Layer normalization for stabilizing training
        # Applied after each sub-layer (attention and feed-forward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Process input through self-attention and feed-forward network.
        
        Args:
            x: Input tensor, shape [batch_size, seq_length, d_model]
            mask: Optional attention mask, shape [batch_size, 1, 1, seq_length]
            
        Returns:
            Processed tensor, shape [batch_size, seq_length, d_model]
        """
        # Self-attention sub-layer
        # Each token attends to all tokens in the sequence
        attn_output = self.self_attn(x, x, x, mask)
        
        # Residual connection and layer normalization
        # This helps with gradient flow and training stability
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward sub-layer
        # Each token is processed independently
        ff_output = self.feed_forward(x)
        
        # Another residual connection and layer normalization
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer with masked self-attention, cross-attention, and feed-forward network.
    
    Each decoder layer has three sub-layers:
    1. Masked self-attention: allows tokens to attend to previous tokens only
    2. Cross-attention: allows tokens to attend to all encoder outputs
    3. Feed-forward network: processes each token individually
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize decoder layer.
        
        Args:
            d_model: Dimension of model (embedding dimension)
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
        """
        super().__init__()
        # Multi-head attention for masked self-attention
        # This prevents tokens from attending to future positions
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Multi-head attention for cross-attention with encoder output
        # This connects the decoder to the encoder
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise feed-forward network
        # This processes each token position independently
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),        # Expand dimension
            nn.ReLU(),                       # Apply non-linearity
            nn.Dropout(dropout),             # Apply dropout for regularization
            nn.Linear(d_ff, d_model)         # Project back to original dimension
        )
        
        # Layer normalization for stabilizing training
        # Applied after each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        Process input through masked self-attention, cross-attention, and feed-forward network.
        
        Args:
            x: Decoder input tensor, shape [batch_size, tgt_seq_length, d_model]
            memory: Encoder output tensor, shape [batch_size, src_seq_length, d_model]
            tgt_mask: Mask for target sequence (prevents looking ahead),
                     shape [batch_size, 1, tgt_seq_length, tgt_seq_length]
            memory_mask: Mask for padding in encoder output,
                        shape [batch_size, 1, 1, src_seq_length]
            
        Returns:
            Processed tensor, shape [batch_size, tgt_seq_length, d_model]
            Cross-attention weights for visualization
        """
        # Masked self-attention sub-layer
        # Each token can only attend to previous tokens (masked)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        
        # Residual connection and layer normalization
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention sub-layer
        # Decoder tokens attend to encoder output (memory)
        # Query comes from decoder, Key and Value from encoder
        attn_output, attn_weights = self.cross_attn(x, memory, memory, memory_mask, return_attention=True)
        
        # Residual connection and layer normalization
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward sub-layer
        # Each token is processed independently
        ff_output = self.feed_forward(x)
        
        # Final residual connection and layer normalization
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    This allows the model to jointly attend to information from different
    representation subspaces at different positions. It splits the embedding
    dimension into multiple heads, applies attention in each head independently,
    then concatenates the results.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Dimension of model (embedding dimension)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model        # Embedding dimension
        self.num_heads = num_heads    # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head
        
        # Linear projections for Query, Key, Value
        # These project the input embeddings to query, key, and value vectors
        self.wq = nn.Linear(d_model, d_model)  # Query projection
        self.wk = nn.Linear(d_model, d_model)  # Key projection
        self.wv = nn.Linear(d_model, d_model)  # Value projection
        
        # Output projection
        # This combines the outputs from all heads back to original dimension
        self.wo = nn.Linear(d_model, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor, shape [batch_size, q_len, d_model]
            key: Key tensor, shape [batch_size, k_len, d_model]
            value: Value tensor, shape [batch_size, v_len, d_model]
            mask: Optional attention mask, shape depends on usage
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor, shape [batch_size, q_len, d_model]
            Optionally, attention weights
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        # Shape transformation: [batch_size, seq_len, d_model] -> 
        # [batch_size, num_heads, seq_len, d_k]
        q = self.wq(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # 1. Matrix multiply Q and K^T
        # Shape: [batch_size, num_heads, q_len, d_k] x [batch_size, num_heads, d_k, k_len]
        # -> [batch_size, num_heads, q_len, k_len]
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # 2. Scale by sqrt(d_k)
        # This prevents extremely small gradients in softmax for large d_k
        scores = scores / math.sqrt(self.d_k)
        
        # 3. Apply mask if provided
        # The mask sets certain positions to a large negative value
        # so they contribute ~0 after softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Apply softmax to get attention weights
        # Shape: [batch_size, num_heads, q_len, k_len]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. Apply attention weights to values
        # Shape: [batch_size, num_heads, q_len, k_len] x [batch_size, num_heads, v_len, d_k]
        # -> [batch_size, num_heads, q_len, d_k]
        output = torch.matmul(attn_weights, v)
        
        # Reshape and apply final projection
        # Shape: [batch_size, num_heads, q_len, d_k] -> [batch_size, q_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.wo(output)
        
        if return_attention:
            return output, attn_weights
        return output


# Utility functions

def create_padding_mask(seq):
    """
    Creates mask for padding tokens (zeros).
    
    This mask ensures that the model doesn't pay attention to padding tokens,
    which don't contain any meaningful information.
    
    Args:
        seq: Sequence tensor, shape [batch_size, seq_length]
        
    Returns:
        Mask tensor, shape [batch_size, 1, 1, seq_length]
        Values are 0 for positions to be masked (padding tokens),
        and 1 for positions to attend to.
    """
    # Create mask where 1 indicates padding tokens (0 in the sequence)
    # Shape: [batch_size, seq_length]
    mask = (seq == 0).unsqueeze(1).unsqueeze(2)
    
    # The extra dimensions are for broadcasting with attention scores
    # Shape: [batch_size, 1, 1, seq_length]
    return mask


def create_look_ahead_mask(size):
    """
    Creates mask to prevent attention to future tokens.
    
    This is used in the decoder to ensure that predictions for a position
    can only depend on known outputs at positions before it.
    
    Args:
        size: Size of the square mask (target sequence length)
        
    Returns:
        Boolean mask tensor, shape [size, size]
        True values indicate positions that should be masked out.
    """
    # Create an upper triangular matrix (1s in the upper triangle)
    # This creates a mask where future positions (upper triangle) are masked
    # Shape: [size, size]
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    
    return mask


def create_masks(src, tgt):
    """
    Creates all necessary masks for training the transformer.
    
    Args:
        src: Source sequence tensor, shape [batch_size, src_seq_length]
        tgt: Target sequence tensor, shape [batch_size, tgt_seq_length]
        
    Returns:
        src_mask: Mask for padding tokens in source
        tgt_mask: Combined mask for target sequence (look-ahead + padding)
        memory_mask: Mask for padding tokens in source (for cross-attention)
    """
    # Create padding mask for source sequence
    # This masks out padding tokens in the source
    src_mask = create_padding_mask(src)
    
    # Create padding mask for target sequence
    # This masks out padding tokens in the target
    tgt_padding_mask = create_padding_mask(tgt)
    
    # Create look-ahead mask for target sequence
    # This prevents attending to future tokens in the target
    tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1)).to(tgt.device)
    
    # Combine padding and look-ahead masks for target
    # A position is masked if it's either padding OR a future position
    # Shape: [batch_size, 1, tgt_seq_length, tgt_seq_length]
    tgt_mask = tgt_padding_mask | tgt_look_ahead_mask.unsqueeze(0)
    
    # The memory mask is the same as the source mask
    # It's used in the decoder's cross-attention to mask padding in encoder output
    return src_mask, tgt_mask, src_mask


# Training function

def train_step(model, optimizer, criterion, src, tgt):
    """
    Performs one training step using teacher forcing.
    
    In teacher forcing, the model receives the correct previous token
    as input at each step, rather than its own prediction.
    
    Args:
        model: Transformer model
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (e.g., CrossEntropyLoss)
        src: Source sequence tensor, shape [batch_size, src_seq_length]
        tgt: Target sequence tensor, shape [batch_size, tgt_seq_length]
        
    Returns:
        Loss value for this batch
    """
    # Create masks for source and target sequences
    src_mask, tgt_mask, memory_mask = create_masks(src, tgt)
    
    # Prepare target for teacher forcing (shift right)
    # Input to decoder: all tokens except the last one
    # Ground truth for prediction: all tokens except the first one (which is usually a start token)
    tgt_input = tgt[:, :-1]     # Remove last token
    tgt_output = tgt[:, 1:]     # Remove first token
    
    # Adjust mask for the shifted target
    tgt_mask = tgt_mask[:, :, :-1, :-1]
    
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass
    # Get model predictions for the target sequence
    outputs, _ = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
    
    # Calculate loss
    # Reshape outputs and target for the loss function
    # From [batch_size, seq_length, vocab_size] to [batch_size*seq_length, vocab_size]
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
    
    # Backward pass and optimization
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    # This sets a maximum value for the gradient norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update model parameters
    optimizer.step()
    
    return loss.item()


# Inference function

def generate_sequence(model, src, max_length=50, start_token=1, end_token=2):
    """
    Generates a sequence using the trained model.
    
    Unlike training, during inference the model uses its own predictions
    as input for the next step (autoregressive generation).
    
    Args:
        model: Transformer model
        src: Source sequence tensor, shape [batch_size, src_seq_length]
        max_length: Maximum length of generated sequence
        start_token: ID of the token to start generation
        end_token: ID of the token that signals sequence end
        
    Returns:
        Generated sequence tensor, shape [batch_size, generated_length]
    """
    # Create source mask for padding tokens
    src_mask = create_padding_mask(src)
    
    # Encode the source sequence
    # This creates contextualized representations of the input
    memory = model.encode(src, src_mask)
    
    # Initialize with start token
    # Shape: [batch_size, 1]
    ys = torch.ones(src.size(0), 1).fill_(start_token).type_as(src).long()
    
    # Generate tokens one by one
    for i in range(max_length - 1):
        # Create target mask
        # This prevents attention to future positions
        tgt_mask = create_look_ahead_mask(ys.size(1)).type_as(src).unsqueeze(0)
        
        # Decode current sequence
        out, _ = model.decode(ys, memory, tgt_mask, src_mask)
        
        # Get prediction for next token (last position)
        # Shape: [batch_size, vocab_size]
        prob = model.final_out(out[:, -1])
        
        # Get most likely token
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)  # Add sequence dimension
        
        # Add predicted token to output sequence
        ys = torch.cat([ys, next_word], dim=1)
        
        # Stop if end token is generated
        if next_word.item() == end_token:
            break
            
    return ys


# Example usage

def test_transformer():
    """
    Test the transformer with sample data.
    
    This function creates a small transformer model and tests it with
    random input data to verify that everything is working correctly.
    """
    # Model parameters
    src_vocab_size = 5000   # Size of source vocabulary
    tgt_vocab_size = 5000   # Size of target vocabulary
    d_model = 256           # Embedding dimension
    num_heads = 8           # Number of attention heads
    num_layers = 3          # Number of encoder/decoder layers
    d_ff = 512              # Dimension of feed-forward network
    
    print("Creating transformer model...")
    # Create model with specified parameters
    model = Transformer(
        src_vocab_size, 
        tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )
    
    print("Generating sample data...")
    # Create random input data to test the model
    batch_size = 2          # Number of sequences in batch
    src_len = 10            # Length of source sequences
    tgt_len = 12            # Length of target sequences
    
    # Random token IDs for source and target
    src = torch.randint(1, 100, (batch_size, src_len))
    tgt = torch.randint(1, 100, (batch_size, tgt_len))
    
    print("Creating masks...")
    # Create necessary attention masks
    src_mask, tgt_mask, memory_mask = create_masks(src, tgt)
    
    print("Running forward pass...")
    # Forward pass through the model
    output, _ = model(src, tgt, src_mask, tgt_mask, memory_mask)
    
    # Print information about model and shapes
    print(f"Input shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, output


if __name__ == "__main__":
    model, output = test_transformer()