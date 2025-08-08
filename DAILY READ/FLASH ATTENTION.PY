"""
FLASHATTENTION ARCHITECTURE: COMPLETE FLOW VISUALIZATION

┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                STANDARD ATTENTION VS FLASHATTENTION                              │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
    ┌───────────────────────────────────────┐   │   ┌───────────────────────────────────────────┐
    │       STANDARD ATTENTION FLOW         │   │   │          FLASHATTENTION FLOW              │
    │                                       │   │   │                                           │
    │  ┌─────────────┐    ┌─────────────┐   │   │   │  ┌─────────────┐    ┌─────────────┐      │
    │  │ Query (Q)   │    │ Key (K)     │   │   │   │  │ Query (Q)   │    │ Key (K)     │      │
    │  └──────┬──────┘    └──────┬──────┘   │   │   │  └──────┬──────┘    └──────┬──────┘      │
    │         │                  │          │   │   │         │                  │             │
    │         └────────┬─────────┘          │   │   │         │                  │             │
    │                  │                    │   │   │         ▼                  ▼             │
    │                  ▼                    │   │   │  ┌─────────────────────────────────┐     │
    │  ┌─────────────────────────────┐      │   │   │  │    BLOCK-WISE PROCESSING       │     │
    │  │ QK^T (Attention Scores)     │      │   │   │  │                               │     │
    │  └──────────────┬──────────────┘      │   │   │  │ • Load Q, K, V blocks to SRAM │     │
    │                 │                     │   │   │  │ • Compute attention for blocks │     │
    │                 ▼                     │   │   │  │ • Accumulate results on-the-fly│     │
    │  ┌─────────────────────────────┐      │   │   │  └─────────────────┬─────────────┘     │
    │  │ Scale (÷ sqrt(d_k))         │      │   │   │                    │                   │
    │  └──────────────┬──────────────┘      │   │   │                    │                   │
    │                 │                     │   │   │                    │                   │
    │                 ▼                     │   │   │                    │                   │
    │  ┌─────────────────────────────┐      │   │   │                    │                   │
    │  │ Softmax                     │      │   │   │                    │                   │
    │  └──────────────┬──────────────┘      │   │   │                    │                   │
    │                 │                     │   │   │                    │                   │
    │                 ▼                     │   │   │                    │                   │
    │  ┌─────────────┐    ┌─────────────┐   │   │   │                    │                   │
    │  │ Attention   │    │ Value (V)   │   │   │   │                    │                   │
    │  │ Weights     │    │             │   │   │   │                    │                   │
    │  └──────┬──────┘    └──────┬──────┘   │   │   │                    │                   │
    │         │                  │          │   │   │                    │                   │
    │         └────────┬─────────┘          │   │   │                    │                   │
    │                  │                    │   │   │                    │                   │
    │                  ▼                    │   │   │                    │                   │
    │  ┌─────────────────────────────┐      │   │   │                    │                   │
    │  │ Attention Output (A·V)      │      │   │   │                    │                   │
    │  └─────────────────────────────┘      │   │   │                    ▼                   │
    │                                       │   │   │  ┌─────────────────────────────────┐   │
    │  MEMORY COMPLEXITY: O(N²)             │   │   │  │      ATTENTION OUTPUT          │   │
    │  All intermediate matrices are stored │   │   │  └─────────────────────────────────┘   │
    │                                       │   │   │                                        │
    │                                       │   │   │  MEMORY COMPLEXITY: O(N)               │
    │                                       │   │   │  No large intermediate matrices stored │
    └───────────────────────────────────────┘   │   └────────────────────────────────────────┘
                                                │
    ┌───────────────────────────────────────────┴───────────────────────────────────────────────┐
    │                                FLASHATTENTION OPTIMIZATIONS                                │
    │                                                                                           │
    │  1. TILING: Break Q, K, V into blocks that fit in fast SRAM memory                        │
    │                                                                                           │
    │  2. RECOMPUTATION: Instead of storing O(N²) attention matrix, recompute when needed       │
    │                                                                                           │
    │  3. SOFTMAX TRICK: Use math identity to compute softmax in a numerically stable way       │
    │     while processing blocks                                                              │
    │                                                                                           │
    │  4. FUSED KERNELS: Combine multiple operations into single GPU kernel for speed           │
    │                                                                                           │
    │  RESULT: O(N) memory usage instead of O(N²), much faster for long sequences               │
    └───────────────────────────────────────────────────────────────────────────────────────────┘

FLASHATTENTION ALGORITHM OVERVIEW:

1. Break Q, K, V matrices into blocks that fit in fast SRAM
2. Process attention block by block to avoid storing full attention matrix
3. Maintain running statistics for softmax normalization
4. Accumulate results on-the-fly with mathematical tricks
5. Achieve same output as standard attention with much less memory

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(nn.Module):
    """
    FlashAttention implementation for efficient attention computation.
    
    This is a simplified implementation of the FlashAttention algorithm
    introduced in "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    (Dao et al., 2022). The actual implementation requires CUDA kernels for maximum
    efficiency, but this version demonstrates the key algorithmic ideas.
    
    The main optimizations are:
    1. Block-wise processing to leverage fast SRAM memory
    2. Avoiding materialization of the full attention matrix
    3. On-the-fly softmax computation with mathematical tricks
    """
    def __init__(self, dropout_prob=0.0, causal=False):
        """
        Initialize FlashAttention module.
        
        Args:
            dropout_prob: Dropout probability for attention weights
            causal: Whether to use causal attention masking (for decoder-only models)
        """
        super().__init__()
        self.dropout_prob = dropout_prob
        self.causal = causal
        
    def forward(self, q, k, v, attn_mask=None):
        """
        Compute attention with the FlashAttention algorithm.
        
        In a real implementation, this would call a CUDA kernel. Here we simulate
        the block-wise computation in PyTorch for clarity.
        
        Args:
            q: Query tensor of shape [batch_size, seq_length, num_heads, head_dim]
            k: Key tensor of shape [batch_size, seq_length, num_heads, head_dim]
            v: Value tensor of shape [batch_size, seq_length, num_heads, head_dim]
            attn_mask: Optional attention mask
            
        Returns:
            Attention output tensor of shape [batch_size, seq_length, num_heads, head_dim]
        """
        # Get dimensions
        batch_size, seqlen_q, num_heads, head_dim = q.shape
        _, seqlen_k, _, _ = k.shape
        
        # Scale factor for attention scores
        scale_factor = 1.0 / math.sqrt(head_dim)
        
        # Determine block sizes based on sequence length
        # In a real implementation, this would be chosen based on hardware SRAM size
        # Here we use a heuristic
        block_size_q = min(seqlen_q, 128)
        block_size_k = min(seqlen_k, 128)
        
        # Initialize output tensor
        output = torch.zeros_like(q)
        
        # Initialize softmax normalization factors
        # m_i: max attention score seen so far for each query
        # l_i: sum of exp(attention scores) seen so far for each query
        m = torch.ones((batch_size, num_heads, seqlen_q)) * -float('inf')
        l = torch.zeros((batch_size, num_heads, seqlen_q))
        
        # Process blocks of queries
        for start_q in range(0, seqlen_q, block_size_q):
            end_q = min(start_q + block_size_q, seqlen_q)
            q_block = q[:, start_q:end_q, :, :]  # [batch_size, block_size_q, num_heads, head_dim]
            
            # Get current softmax statistics for this query block
            m_block = m[:, :, start_q:end_q]  # [batch_size, num_heads, block_size_q]
            l_block = l[:, :, start_q:end_q]  # [batch_size, num_heads, block_size_q]
            
            # Initialize output accumulator for this query block
            o_block = torch.zeros_like(q_block)  # [batch_size, block_size_q, num_heads, head_dim]
            
            # Process blocks of keys/values
            for start_k in range(0, seqlen_k, block_size_k):
                end_k = min(start_k + block_size_k, seqlen_k)
                k_block = k[:, start_k:end_k, :, :]  # [batch_size, block_size_k, num_heads, head_dim]
                v_block = v[:, start_k:end_k, :, :]  # [batch_size, block_size_k, num_heads, head_dim]
                
                # Compute attention scores for this block
                # Reshape for batch matrix multiplication
                q_block_reshaped = q_block.transpose(1, 2)  # [batch_size, num_heads, block_size_q, head_dim]
                k_block_reshaped = k_block.transpose(1, 2).transpose(2, 3)  # [batch_size, num_heads, head_dim, block_size_k]
                
                # Compute attention scores: [batch_size, num_heads, block_size_q, block_size_k]
                attn_scores = torch.matmul(q_block_reshaped, k_block_reshaped) * scale_factor
                
                # Apply causal mask if needed
                if self.causal and start_q + block_size_q > start_k:
                    # Create causal mask for this block
                    causal_mask = torch.ones(block_size_q, block_size_k, dtype=torch.bool, device=q.device)
                    for i in range(block_size_q):
                        q_idx = start_q + i
                        for j in range(block_size_k):
                            k_idx = start_k + j
                            if q_idx < k_idx:  # Don't allow attention to future tokens
                                causal_mask[i, j] = False
                    
                    # Apply causal mask to attention scores
                    mask = causal_mask.view(1, 1, block_size_q, block_size_k)
                    attn_scores = attn_scores.masked_fill(~mask, -float('inf'))
                
                # Apply additional attention mask if provided
                if attn_mask is not None:
                    block_mask = attn_mask[:, start_q:end_q, start_k:end_k].unsqueeze(1)  # Add head dimension
                    attn_scores = attn_scores + block_mask
                
                # Update softmax statistics
                m_block_new = torch.maximum(
                    m_block.unsqueeze(-1), 
                    attn_scores.max(dim=-1, keepdim=True)[0]
                )  # [batch_size, num_heads, block_size_q, 1]
                
                # Compute scaling factors for re-normalization
                # When max changes, we need to rescale the accumulated sum
                m_block_diff = m_block_new - m_block.unsqueeze(-1)
                scaling_factor = torch.exp(-m_block_diff)
                
                # Update sum of exponentiated attention scores with rescaled old sum
                l_block_new = scaling_factor * l_block.unsqueeze(-1) + torch.sum(
                    torch.exp(attn_scores - m_block_new), dim=-1, keepdim=True
                )  # [batch_size, num_heads, block_size_q, 1]
                
                # Compute attention weights for this block
                # Using the updated softmax statistics
                attn_weights = torch.exp(attn_scores - m_block_new) / l_block_new
                
                # Apply dropout if probability is greater than 0
                if self.dropout_prob > 0.0 and self.training:
                    dropout_mask = torch.bernoulli(
                        torch.ones_like(attn_weights) * (1 - self.dropout_prob)
                    ).to(attn_weights.dtype) / (1 - self.dropout_prob)
                    attn_weights = attn_weights * dropout_mask
                
                # Compute weighted sum of values for this block
                v_block_reshaped = v_block.transpose(1, 2)  # [batch_size, num_heads, block_size_k, head_dim]
                o_block_update = torch.matmul(attn_weights, v_block_reshaped)  # [batch_size, num_heads, block_size_q, head_dim]
                
                # Rescale accumulated output with new softmax statistics
                o_block_scaled = (
                    o_block.transpose(1, 2) * scaling_factor.unsqueeze(-1)
                )  # [batch_size, num_heads, block_size_q, head_dim]
                
                # Update output accumulator
                o_block = (o_block_scaled + o_block_update).transpose(1, 2)
                
                # Update softmax statistics for next iteration
                m_block = m_block_new.squeeze(-1)
                l_block = l_block_new.squeeze(-1)
            
            # Update global softmax statistics
            m[:, :, start_q:end_q] = m_block
            l[:, :, start_q:end_q] = l_block
            
            # Write block output to global output tensor
            output[:, start_q:end_q, :, :] = o_block
        
        return output


class FlashMHA(nn.Module):
    """
    Multi-head attention using FlashAttention.
    
    This combines the standard query, key, value projections with the 
    FlashAttention algorithm for efficient attention computation.
    """
    def __init__(self, hidden_size, num_heads, dropout_prob=0.0, causal=False):
        """
        Initialize multi-head attention with FlashAttention.
        
        Args:
            hidden_size: Dimension of input and output
            num_heads: Number of attention heads
            dropout_prob: Dropout probability for attention weights
            causal: Whether to use causal attention masking
        """
        super().__init__()
        
        # Ensure hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by the number of attention heads {num_heads}"
            )
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # QKV projection (using a single matrix for efficiency)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # FlashAttention implementation
        self.flash_attn = FlashAttention(dropout_prob, causal)
        
        # Dropout for output
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Apply multi-head attention with FlashAttention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            
        Returns:
            Attention output tensor of same shape as input
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project input to query, key, value using a single matrix
        qkv = self.qkv_proj(hidden_states)
        
        # Reshape and split into query, key, value
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply FlashAttention
        attn_output = self.flash_attn(q, k, v, attention_mask)
        
        # Reshape attention output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size
        )
        
        # Apply output projection and dropout
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        
        return output


class FlashTransformerBlock(nn.Module):
    """
    Transformer block using FlashAttention.
    
    This is compatible with both BERT-style (bidirectional) and GPT-style (causal)
    transformer architectures.
    """
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_prob=0.1,
                layer_norm_eps=1e-12, causal=False, pre_ln=True):
        """
        Initialize transformer block with FlashAttention.
        
        Args:
            hidden_size: Dimension of input and output
            num_heads: Number of attention heads
            intermediate_size: Dimension of feed-forward network
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization stability
            causal: Whether to use causal attention masking
            pre_ln: Whether to use Pre-LN architecture (LayerNorm before attention/FFN)
                    rather than Post-LN (original Transformer)
        """
        super().__init__()
        
        # Multi-head attention with FlashAttention
        self.attn = FlashMHA(hidden_size, num_heads, dropout_prob, causal)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout_prob)
        
        # Architecture type
        self.pre_ln = pre_ln
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Process input through transformer block with FlashAttention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            
        Returns:
            Processed tensor after attention and feed-forward
        """
        # Attention block
        if self.pre_ln:
            # Pre-LN architecture (better training stability)
            # Layer normalization before attention
            norm_output = self.ln1(hidden_states)
            # Apply attention
            attn_output = self.attn(norm_output, attention_mask)
            # Residual connection
            hidden_states = hidden_states + attn_output
        else:
            # Post-LN architecture (original Transformer)
            # Apply attention
            attn_output = self.attn(hidden_states, attention_mask)
            # Residual connection and layer normalization
            hidden_states = self.ln1(hidden_states + attn_output)
        
        # Feed-forward block
        if self.pre_ln:
            # Pre-LN architecture
            # Layer normalization before feed-forward
            norm_output = self.ln2(hidden_states)
            # Apply feed-forward
            ffn_output = self.ffn(norm_output)
            # Residual connection
            hidden_states = hidden_states + ffn_output
        else:
            # Post-LN architecture
            # Apply feed-forward
            ffn_output = self.ffn(hidden_states)
            # Residual connection and layer normalization
            hidden_states = self.ln2(hidden_states + ffn_output)
        
        return hidden_states


class FlashBertEncoder(nn.Module):
    """
    BERT-style encoder using FlashAttention.
    
    This is a bidirectional encoder that allows each token to attend to all
    other tokens in the sequence.
    """
    def __init__(self, hidden_size, num_heads, intermediate_size, num_hidden_layers,
                dropout_prob=0.1, layer_norm_eps=1e-12, pre_ln=True):
        """
        Initialize BERT encoder with FlashAttention.
        
        Args:
            hidden_size: Dimension of input and output
            num_heads: Number of attention heads
            intermediate_size: Dimension of feed-forward network
            num_hidden_layers: Number of transformer layers
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization stability
            pre_ln: Whether to use Pre-LN architecture
        """
        super().__init__()
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            FlashTransformerBlock(
                hidden_size, num_heads, intermediate_size, dropout_prob,
                layer_norm_eps, causal=False, pre_ln=pre_ln
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final layer normalization for Pre-LN architecture
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if pre_ln else None
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Process input through the encoder stack.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            
        Returns:
            Encoded tensor after all transformer layers
        """
        # Process through each transformer block
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Apply final layer normalization for Pre-LN architecture
        if self.ln_f is not None:
            hidden_states = self.ln_f(hidden_states)
        
        return hidden_states


class FlashGPTDecoder(nn.Module):
    """
    GPT-style decoder using FlashAttention.
    
    This is a causal decoder where each token can only attend to itself
    and previous tokens.
    """
    def __init__(self, hidden_size, num_heads, intermediate_size, num_hidden_layers,
                dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize GPT decoder with FlashAttention.
        
        Args:
            hidden_size: Dimension of input and output
            num_heads: Number of attention heads
            intermediate_size: Dimension of feed-forward network
            num_hidden_layers: Number of transformer layers
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization stability
        """
        super().__init__()
        
        # Stack of transformer blocks with causal attention
        self.layers = nn.ModuleList([
            FlashTransformerBlock(
                hidden_size, num_heads, intermediate_size, dropout_prob,
                layer_norm_eps, causal=True, pre_ln=True  # GPT uses Pre-LN
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Process input through the decoder stack.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            
        Returns:
            Decoded tensor after all transformer layers
        """
        # Process through each transformer block
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Apply final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states


class FlashCrossAttention(nn.Module):
    """
    Cross-attention using FlashAttention.
    
    This is used in encoder-decoder transformer architectures where the decoder
    attends to the encoder's outputs.
    """
    def __init__(self, hidden_size, num_heads, dropout_prob=0.0):
        """
        Initialize cross-attention with FlashAttention.
        
        Args:
            hidden_size: Dimension of input and output
            num_heads: Number of attention heads
            dropout_prob: Dropout probability for attention weights
        """
        super().__init__()
        
        # Ensure hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by the number of attention heads {num_heads}"
            )
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query projection (from decoder)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        
        # Key and value projections (from encoder)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # FlashAttention implementation (non-causal)
        self.flash_attn = FlashAttention(dropout_prob, causal=False)
        
        # Dropout for output
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, decoder_states, encoder_states, attention_mask=None):
        """
        Apply cross-attention with FlashAttention.
        
        Args:
            decoder_states: Decoder hidden states (for query)
                           Shape: [batch_size, tgt_length, hidden_size]
            encoder_states: Encoder hidden states (for key and value)
                           Shape: [batch_size, src_length, hidden_size]
            attention_mask: Optional attention mask tensor
            
        Returns:
            Cross-attention output tensor
        """
        batch_size, tgt_length, _ = decoder_states.shape
        _, src_length, _ = encoder_states.shape
        
        # Project inputs to query, key, value
        q = self.q_proj(decoder_states)  # [batch_size, tgt_length, hidden_size]
        k = self.k_proj(encoder_states)  # [batch_size, src_length, hidden_size]
        v = self.v_proj(encoder_states)  # [batch_size, src_length, hidden_size]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, tgt_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, src_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, src_length, self.num_heads, self.head_dim)
        
        # Apply FlashAttention
        attn_output = self.flash_attn(q, k, v, attention_mask)
        
        # Reshape attention output
        attn_output = attn_output.reshape(batch_size, tgt_length, self.hidden_size)
        
        # Apply output projection and dropout
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        
        return output


class FlashEncoderDecoderBlock(nn.Module):
    """
    Transformer decoder block with self-attention, cross-attention, and feed-forward,
    all using FlashAttention.
    
    This is used in encoder-decoder transformer architectures like T5 or BART.
    """
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_prob=0.1,
                layer_norm_eps=1e-12, pre_ln=True):
        """
        Initialize encoder-decoder block with FlashAttention.
        
        Args:
            hidden_size: Dimension of input and output
            num_heads: Number of attention heads
            intermediate_size: Dimension of feed-forward network
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization stability
            pre_ln: Whether to use Pre-LN architecture
        """
        super().__init__()
        
        # Self-attention (causal, decoder-side)
        self.self_attn = FlashMHA(hidden_size, num_heads, dropout_prob, causal=True)
        
        # Cross-attention (attends to encoder outputs)
        self.cross_attn = FlashCrossAttention(hidden_size, num_heads, dropout_prob)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln3 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Architecture type
        self.pre_ln = pre_ln
        
    def forward(self, decoder_states, encoder_states, self_attn_mask=None, cross_attn_mask=None):
        """
        Process decoder input with self-attention, cross-attention, and feed-forward.
        
        Args:
            decoder_states: Decoder hidden states
                           Shape: [batch_size, tgt_length, hidden_size]
            encoder_states: Encoder hidden states
                           Shape: [batch_size, src_length, hidden_size]
            self_attn_mask: Mask for self-attention (usually causal)
            cross_attn_mask: Mask for cross-attention (usually for padding)
            
        Returns:
            Processed tensor after all sub-layers
        """
        # Self-attention block
        if self.pre_ln:
            # Pre-LN architecture
            norm_output = self.ln1(decoder_states)
            attn_output = self.self_attn(norm_output, self_attn_mask)
            decoder_states = decoder_states + attn_output
        else:
            # Post-LN architecture
            attn_output = self.self_attn(decoder_states, self_attn_mask)
            decoder_states = self.ln1(decoder_states + attn_output)
        
        # Cross-attention block
        if self.pre_ln:
            # Pre-LN architecture
            norm_output = self.ln2(decoder_states)
            cross_output = self.cross_attn(norm_output, encoder_states, cross_attn_mask)
            decoder_states = decoder_states + cross_output
        else:
            # Post-LN architecture
            cross_output = self.cross_attn(decoder_states, encoder_states, cross_attn_mask)
            decoder_states = self.ln2(decoder_states + cross_output)
        
        # Feed-forward block
        if self.pre_ln:
            # Pre-LN architecture
            norm_output = self.ln3(decoder_states)
            ffn_output = self.ffn(norm_output)
            decoder_states = decoder_states + ffn_output
        else:
            # Post-LN architecture
            ffn_output = self.ffn(decoder_states)
            decoder_states = self.ln3(decoder_states + ffn_output)
        
        return decoder_states


# Utility functions for block-sparse attention

def generate_local_block_sparse_mask(seq_length, block_size, window_size):
    """
    Generate a local block-sparse attention mask.
    
    This creates a mask where each token attends only to tokens within a local window.
    The mask is block-sparse, meaning attention is computed in blocks for efficiency.
    
    Args:
        seq_length: Sequence length
        block_size: Size of attention blocks
        window_size: Size of the local attention window
        
    Returns:
        Block-sparse attention mask
    """
    # Number of blocks
    num_blocks = (seq_length + block_size - 1) // block_size
    
    # Create block mask
    block_mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool)
    
    # Set blocks within window to True
    window_blocks = (window_size + block_size - 1) // block_size
    for i in range(num_blocks):
        start = max(0, i - window_blocks)
        end = min(num_blocks, i + window_blocks + 1)
        block_mask[i, start:end] = True
    
    # Expand block mask to token level
    mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
    for i in range(num_blocks):
        for j in range(num_blocks):
            if block_mask[i, j]:
                # Block start and end indices
                i_start = i * block_size
                i_end = min(seq_length, (i + 1) * block_size)
                j_start = j * block_size
                j_end = min(seq_length, (j + 1) * block_size)
                
                # Set block to True
                mask[i_start:i_end, j_start:j_end] = True
    
    return mask


class BlockSparseFlashAttention(nn.Module):
    """
    Block-sparse version of FlashAttention.
    
    This further improves efficiency by only computing attention for certain blocks,
    such as local windows or global tokens, rather than the full attention matrix.
    """
    def __init__(self, block_size=64, sparsity_config='local', local_window_size=128,
                dropout_prob=0.0, causal=False):
        """
        Initialize block-sparse FlashAttention.
        
        Args:
            block_size: Size of attention blocks
            sparsity_config: Type of sparsity pattern ('local', 'global', 'random')
            local_window_size: Size of local attention window (if applicable)
            dropout_prob: Dropout probability for attention weights
            causal: Whether to use causal attention masking
        """
        super().__init__()
        self.block_size = block_size
        self.sparsity_config = sparsity_config
        self.local_window_size = local_window_size
        self.dropout_prob = dropout_prob
        self.causal = causal
        
        # Standard FlashAttention for computing attention within blocks
        self.flash_attn = FlashAttention(dropout_prob, causal)
        
    def forward(self, q, k, v, attn_mask=None):
        """
        Compute block-sparse attention with FlashAttention.
        
        Args:
            q: Query tensor of shape [batch_size, seq_length, num_heads, head_dim]
            k: Key tensor of shape [batch_size, seq_length, num_heads, head_dim]
            v: Value tensor of shape [batch_size, seq_length, num_heads, head_dim]
            attn_mask: Optional attention mask
            
        Returns:
            Attention output tensor of shape [batch_size, seq_length, num_heads, head_dim]
        """
        # Get dimensions
        batch_size, seqlen_q, num_heads, head_dim = q.shape
        _, seqlen_k, _, _ = k.shape
        
        # Generate block-sparse mask based on sparsity configuration
        if self.sparsity_config == 'local':
            # Local window attention
            block_sparse_mask = generate_local_block_sparse_mask(
                seqlen_q, self.block_size, self.local_window_size
            ).to(q.device)
        elif self.sparsity_config == 'global':
            # Global tokens attend to all, others use local attention
            # For simplicity, we'll make the first block global
            block_sparse_mask = generate_local_block_sparse_mask(
                seqlen_q, self.block_size, self.local_window_size
            ).to(q.device)
            # Make first block global
            first_block_size = min(self.block_size, seqlen_q)
            block_sparse_mask[:first_block_size, :] = True
            block_sparse_mask[:, :first_block_size] = True
        else:
            # Default to full attention
            block_sparse_mask = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)
        
        # Combine with provided attention mask if any
        if attn_mask is not None:
            block_sparse_mask = block_sparse_mask & attn_mask.squeeze(0).squeeze(0).bool()
        
        # Initialize output tensor
        output = torch.zeros_like(q)
        
        # Get block indices based on the mask
        # In a real implementation, this would be done more efficiently
        block_indices = []
        num_q_blocks = (seqlen_q + self.block_size - 1) // self.block_size
        num_k_blocks = (seqlen_k + self.block_size - 1) // self.block_size
        
        for i in range(num_q_blocks):
            for j in range(num_k_blocks):
                # Block start and end indices
                q_start = i * self.block_size
                q_end = min(seqlen_q, (i + 1) * self.block_size)
                k_start = j * self.block_size
                k_end = min(seqlen_k, (j + 1) * self.block_size)
                
                # Check if this block is in the sparse mask
                if block_sparse_mask[q_start, k_start]:
                    block_indices.append((q_start, q_end, k_start, k_end))
        
        # Process each active block with FlashAttention
        for q_start, q_end, k_start, k_end in block_indices:
            q_block = q[:, q_start:q_end, :, :]
            k_block = k[:, k_start:k_end, :, :]
            v_block = v[:, k_start:k_end, :, :]
            
            # Apply FlashAttention to this block
            # In a real implementation, we would maintain global softmax normalization
            output_block = self.flash_attn(q_block, k_block, v_block)
            
            # Accumulate results
            # Note: This is a simplification. In a real implementation, we would
            # need to handle the normalization properly across blocks.
            output[:, q_start:q_end, :, :] += output_block
        
        return output


# Example usage with different transformer architectures

def create_flash_bert_model(vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                           num_attention_heads=12, intermediate_size=3072):
    """
    Create a BERT-style model with FlashAttention.
    """
    # Token and position embeddings
    embeddings = nn.Embedding(vocab_size, hidden_size)
    position_embeddings = nn.Embedding(512, hidden_size)
    
    # Layer normalization and dropout
    layer_norm = nn.LayerNorm(hidden_size)
    dropout = nn.Dropout(0.1)
    
    # BERT encoder with FlashAttention
    encoder = FlashBertEncoder(
        hidden_size=hidden_size,
        num_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        dropout_prob=0.1,
        pre_ln=True  # Modern BERT variants often use Pre-LN
    )
    
    # Return components as a dictionary
    return {
        'embeddings': embeddings,
        'position_embeddings': position_embeddings,
        'layer_norm': layer_norm,
        'dropout': dropout,
        'encoder': encoder
    }


def create_flash_gpt_model(vocab_size=50257, hidden_size=768, num_hidden_layers=12,
                          num_attention_heads=12, intermediate_size=3072):
    """
    Create a GPT-style model with FlashAttention.
    """
    # Token and position embeddings
    embeddings = nn.Embedding(vocab_size, hidden_size)
    position_embeddings = nn.Embedding(1024, hidden_size)
    
    # Dropout
    dropout = nn.Dropout(0.1)
    
    # GPT decoder with FlashAttention
    decoder = FlashGPTDecoder(
        hidden_size=hidden_size,
        num_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        dropout_prob=0.1
    )
    
    # Linear head for next token prediction
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    # Return components as a dictionary
    return {
        'embeddings': embeddings,
        'position_embeddings': position_embeddings,
        'dropout': dropout,
        'decoder': decoder,
        'lm_head': lm_head
    }


def test_flash_attention():
    """
    Test FlashAttention with random input data.
    """
    # Parameters
    batch_size = 2
    seq_length = 1024
    hidden_size = 768
    num_heads = 12
    head_dim = hidden_size // num_heads
    
    # Create random inputs
    q = torch.randn(batch_size, seq_length, num_heads, head_dim)
    k = torch.randn(batch_size, seq_length, num_heads, head_dim)
    v = torch.randn(batch_size, seq_length, num_heads, head_dim)
    
    # Test standard attention
    flash_attn = FlashAttention(dropout_prob=0.0)
    
    # Measure time and memory for FlashAttention
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    # Record memory before
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    memory_before = torch.cuda.max_memory_allocated()
    
    # Time the forward pass
    start_time.record()
    output = flash_attn(q, k, v)
    end_time.record()
    
    # Synchronize and get memory usage
    torch.cuda.synchronize()
    memory_after = torch.cuda.max_memory_allocated()
    memory_used = memory_after - memory_before
    
    # Get elapsed time
    elapsed_time = start_time.elapsed_time(end_time)
    
    print(f"FlashAttention test:")
    print(f"Input shapes: Q, K, V: {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Elapsed time: {elapsed_time:.2f} ms")
    print(f"Memory used: {memory_used / 1024**2:.2f} MB")
    
    return output


def compare_standard_vs_flash_attention(seq_length=4096, batch_size=2, num_heads=16, head_dim=64):
    """
    Compare standard attention with FlashAttention for memory usage and speed.
    
    This demonstrates the memory efficiency of FlashAttention for long sequences.
    """
    print(f"Comparing Standard Attention vs FlashAttention")
    print(f"Sequence length: {seq_length}, Batch size: {batch_size}")
    print(f"Attention heads: {num_heads}, Head dimension: {head_dim}")
    
    # Create random inputs
    hidden_size = num_heads * head_dim
    q = torch.randn(batch_size, seq_length, hidden_size).cuda()
    k = torch.randn(batch_size, seq_length, hidden_size).cuda()
    v = torch.randn(batch_size, seq_length, hidden_size).cuda()
    
    # Standard attention function (simplified)
    def standard_attention(q, k, v):
        # Project to multi-head
        batch_size, seq_len, hidden_size = q.shape
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return output
    
    # FlashAttention module
    flash_attn = FlashMHA(hidden_size, num_heads, dropout_prob=0.0).cuda()
    
    # Measure standard attention
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        start_time = time.time()
        standard_output = standard_attention(q, k, v)
        torch.cuda.synchronize()
        
        standard_time = time.time() - start_time
        standard_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"Standard Attention:")
        print(f"  Time: {standard_time*1000:.2f} ms")
        print(f"  Memory: {standard_memory:.2f} MB")
    except RuntimeError as e:
        print(f"Standard Attention: Out of memory - {e}")
        standard_output = None
    
    # Measure FlashAttention
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    start_time = time.time()
    flash_output = flash_attn(q)
    torch.cuda.synchronize()
    
    flash_time = time.time() - start_time
    flash_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"FlashAttention:")
    print(f"  Time: {flash_time*1000:.2f} ms")
    print(f"  Memory: {flash_memory:.2f} MB")
    
    # Compare results if standard attention didn't OOM
    if standard_output is not None:
        max_diff = torch.max(torch.abs(standard_output - flash_output))
        print(f"Maximum difference between outputs: {max_diff:.6f}")
    
    # Calculate improvements
    if standard_output is not None:
        speedup = standard_time / flash_time
        memory_reduction = standard_memory / flash_memory
        print(f"Speedup: {speedup:.2f}x")
        print(f"Memory reduction: {memory_reduction:.2f}x")
    
    return flash_output


if __name__ == "__main__":
    # Test FlashAttention
    output = test_flash_attention()
    
    # On a CUDA-enabled system, you could also compare with standard attention
    # Note: For very long sequences, standard attention may run out of memory
    # compare_standard_vs_flash_attention(seq_length=4096)