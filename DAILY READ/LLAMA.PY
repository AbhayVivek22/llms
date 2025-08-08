"""
LLAMA ARCHITECTURE: COMPLETE FLOW VISUALIZATION

                          INPUT SEQUENCE
                               │
                               ▼
┌─────────────────────────────────────────────┐
│               TOKEN EMBEDDING               │
└───────────────────────┬─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────┐ │
│ │  DECODER BLOCK 1                        │ │
│ │                                         │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │         RMS NORMALIZATION         │  │ │  <-- Different from LayerNorm in BERT/GPT
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │    MULTI-HEAD ATTENTION + RoPE    │  │ │  <-- RoPE = Rotary Position Embedding
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │         RESIDUAL CONNECTION       │  │ │
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │         RMS NORMALIZATION         │  │ │
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │       SwiGLU FEED-FORWARD         │  │ │  <-- SwiGLU instead of GELU/ReLU
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │         RESIDUAL CONNECTION       │  │ │
│ │  └─────────────────┬─────────────────┘  │ │
│ └────────────────────┼─────────────────────┘
│                      │
│                      ▼
│ ┌─────────────────────────────────────────┐
│ │  DECODER BLOCK 2...N                    │
│ └────────────────────┼─────────────────────┘
└──────────────────────┼──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              RMS NORMALIZATION              │  <-- Final normalization
└───────────────────────┬─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│               LINEAR PROJECTION             │  <-- Project to vocabulary size
└───────────────────────┬─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│                   SOFTMAX                   │  <-- Convert to probabilities
└───────────────────────┬─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│              NEXT TOKEN PREDICTION          │
└─────────────────────────────────────────────┘

LLAMA KEY FEATURES:

1. Decoder-Only Architecture - Similar to GPT, autoregressive next-token prediction
2. RMSNorm - Root Mean Square Normalization instead of LayerNorm (more efficient)
3. SwiGLU Activation - Swish-Gated Linear Unit instead of GELU in feed-forward networks
4. Rotary Positional Embeddings (RoPE) - Encodes positions directly in attention mechanism
5. Pre-Normalization - LayerNorm before attention and feed-forward (like modern GPT)
6. Grouped-Query Attention (GQA) - Efficiency improvement between MHA and MQA (Llama 2/3)
7. Larger Context Windows - Trained on sequences of 4K tokens (Llama 2) or more

MULTI-QUERY ATTENTION VS GROUPED-QUERY ATTENTION:

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  STANDARD MHA   │  │  MULTI-QUERY    │  │  GROUPED-QUERY  │
│                 │  │  ATTENTION      │  │  ATTENTION      │
│  Q₁  K₁  V₁     │  │  Q₁      \      │  │  Q₁      \      │
│  │   │   │      │  │  │        \     │  │  │        \     │
│  Q₂  K₂  V₂     │  │  Q₂  K  V       │  │  Q₂  K₁  V₁     │
│  │   │   │      │  │  │   │  │       │  │  │   │   │      │
│  Q₃  K₃  V₃     │  │  Q₃ /   \       │  │  Q₃ /    \      │
│  │   │   │      │  │  │     /        │  │  │       \      │
│  Q₄  K₄  V₄     │  │  Q₄ ___/        │  │  Q₄  K₂  V₂     │
│                 │  │                  │  │                 │
│ #params: 3*d²   │  │ #params: d²+2d  │  │ #params: varies │
└─────────────────┘  └─────────────────┘  └─────────────────┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LlamaConfig:
    """
    Configuration class to store the configuration of a Llama model.
    
    This class holds all hyperparameters for the model architecture, making it
    easy to create different model sizes by just changing the configuration.
    """
    def __init__(
        self,
        vocab_size=32000,           # Size of the vocabulary
        hidden_size=4096,           # Dimension of embeddings and hidden layers
        intermediate_size=11008,    # Dimension of feed-forward network
        num_hidden_layers=32,       # Number of transformer layers
        num_attention_heads=32,     # Number of attention heads
        num_key_value_heads=None,   # Number of key/value heads for GQA (defaults to same as attention heads)
        hidden_act="silu",          # Activation function for feed-forward
        max_position_embeddings=2048,  # Maximum sequence length
        initializer_range=0.02,     # Standard deviation for initializing weights
        rms_norm_eps=1e-6,          # Small constant for RMSNorm stability
        use_cache=True,             # Whether to use KV cache for faster generation
        pad_token_id=0,             # ID of the padding token
        bos_token_id=1,             # ID of the beginning of sequence token
        eos_token_id=2,             # ID of the end of sequence token
        tie_word_embeddings=False,  # Whether to tie input/output embeddings
    ):
        """
        Initialize Llama configuration.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Dimension of embeddings and hidden layers
            intermediate_size: Dimension of feed-forward network
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads for grouped-query attention (GQA)
                                If None, uses the same as num_attention_heads (regular MHA)
            hidden_act: Activation function ("silu" for Llama)
            max_position_embeddings: Maximum sequence length
            initializer_range: Standard deviation for initializing weights
            rms_norm_eps: Small constant for RMSNorm stability
            use_cache: Whether to use KV cache for faster generation
            pad_token_id: ID of the padding token
            bos_token_id: ID of the beginning of sequence token
            eos_token_id: ID of the end of sequence token
            tie_word_embeddings: Whether to tie input/output embeddings
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        # For Grouped-Query Attention (GQA)
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm is a simplified version of LayerNorm that only normalizes by the root
    mean square of the inputs, without centering (subtracting the mean). This makes
    it more computationally efficient while maintaining good performance.
    
    Paper: "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467)
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            hidden_size: Feature dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        """
        Apply RMS normalization.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            Normalized tensor of the same shape
        """
        # Calculate the root mean square along the feature dimension
        # The key difference from LayerNorm is we don't subtract the mean
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # Scale with learned parameters
        return self.weight * hidden_states


def precompute_rotary_embedding(seq_len, dim, theta=10000.0, device=None):
    """
    Precompute the rotary position embeddings.
    
    This creates a tensor of complex numbers used for the rotary position encoding.
    Rotary position embeddings (RoPE) encode absolute positional information by
    rotating vector representations based on position in the sequence.
    
    Args:
        seq_len: Maximum sequence length
        dim: Dimension of the embeddings (must be divisible by 2)
        theta: Base frequency parameter
        device: Device to place the tensor on
        
    Returns:
        Complex tensor with positional frequencies [seq_len, dim//2]
    """
    # Create position indices: 0, 1, 2, ..., seq_len-1
    position = torch.arange(seq_len, device=device).float()
    # Create dimension indices: 0, 2, 4, ..., dim-2
    indices = torch.arange(0, dim, 2, device=device).float()
    
    # The angle is position * (1 / theta^(2i/dim))
    # This creates dimensions with different frequencies
    # Lower dimensions change slowly, higher dimensions change rapidly with position
    freqs = 1.0 / (theta ** (indices / dim))
    
    # Compute angles for each position and dimension pair
    # Shape: [seq_len, dim//2]
    angles = torch.outer(position, freqs)
    
    # Convert to complex numbers: e^(i*theta) = cos(theta) + i*sin(theta)
    # This is a convenient way to store both cos and sin values
    cos_sin = torch.polar(torch.ones_like(angles), angles)
    
    return cos_sin


def apply_rotary_embeddings(q, k, cos_sin, position_ids=None):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Rotary position embeddings rotate each feature dimension pair by a position-dependent
    amount, providing the model with information about token positions without requiring
    separate position embeddings.
    
    Args:
        q: Query tensor of shape [batch_size, seq_length, num_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_length, num_heads, head_dim]
        cos_sin: Precomputed rotary embeddings of shape [seq_len, head_dim//2]
        position_ids: Optional tensor of position indices [batch_size, seq_length]
        
    Returns:
        Rotated query and key tensors
    """
    # Get shapes
    batch_size, seq_length, num_heads, head_dim = q.shape
    
    # If no explicit position IDs are provided, use the default sequential positions
    if position_ids is None:
        position_ids = torch.arange(seq_length, device=q.device).unsqueeze(0)
    
    # Get the appropriate position embeddings for each position
    # Shape: [batch_size, seq_length, head_dim//2]
    cos_sin = cos_sin[position_ids].unsqueeze(2)  # Add head dimension
    
    # Extract real (cos) and imaginary (sin) parts
    # Both have shape [batch_size, seq_length, 1, head_dim//2]
    cos = cos_sin.real
    sin = cos_sin.imag
    
    # Reshape q and k for rotation
    # We need to treat each dimension pair (dim_i, dim_i+1) as a complex number
    # First, split the head dimension into pairs
    q_split = q.reshape(batch_size, seq_length, num_heads, head_dim//2, 2)
    k_split = k.reshape(batch_size, seq_length, num_heads, head_dim//2, 2)
    
    # For each pair, we interpret it as (real, imag) parts of a complex number
    q_real, q_imag = q_split[..., 0], q_split[..., 1]
    k_real, k_imag = k_split[..., 0], k_split[..., 1]
    
    # Apply the rotation (complex multiplication with cos+i*sin)
    # For a complex number a+bi and rotation cos+i*sin:
    # (a+bi)(cos+i*sin) = (a*cos-b*sin) + i(a*sin+b*cos)
    q_out_real = q_real * cos - q_imag * sin
    q_out_imag = q_real * sin + q_imag * cos
    k_out_real = k_real * cos - k_imag * sin
    k_out_imag = k_real * sin + k_imag * cos
    
    # Combine the real and imaginary parts
    q_out = torch.stack([q_out_real, q_out_imag], dim=-1)
    k_out = torch.stack([k_out_real, k_out_imag], dim=-1)
    
    # Reshape back to original shape
    q_out = q_out.reshape(batch_size, seq_length, num_heads, head_dim)
    k_out = k_out.reshape(batch_size, seq_length, num_heads, head_dim)
    
    return q_out, k_out


class LlamaAttention(nn.Module):
    """
    Multi-head attention with rotary position embeddings.
    
    This implements both standard multi-head attention and grouped-query attention (GQA),
    where there are fewer key/value heads than query heads to reduce computation and
    memory requirements while maintaining most of the model quality.
    """
    def __init__(self, config):
        """
        Initialize Llama attention module.
        
        Args:
            config: LlamaConfig instance with model configuration
        """
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # For computing Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary positional embeddings
        self.max_position_embeddings = config.max_position_embeddings
        
        # Initialize a buffer to hold the cos/sin values for RoPE
        # We'll generate this in the first forward pass for the specific sequence length
        self.register_buffer(
            "rotary_emb",
            None,
            persistent=False,
        )
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        """
        Apply attention with rotary position embeddings.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            position_ids: Optional tensor of position indices
            past_key_value: Optional tuple of cached key and value projections
            output_attentions: Whether to return attention weights
            use_cache: Whether to return key/value states for subsequent generation
            
        Returns:
            Attention output tensor
            Optional attention weights
            Optional key/value cache
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project inputs to queries, keys, and values
        query_states = self.q_proj(hidden_states)  # [batch_size, seq_length, num_heads * head_dim]
        key_states = self.k_proj(hidden_states)    # [batch_size, seq_length, num_kv_heads * head_dim]
        value_states = self.v_proj(hidden_states)  # [batch_size, seq_length, num_kv_heads * head_dim]
        
        # Reshape to [batch_size, seq_length, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        
        # If we're using key/value cache, then prepend past keys and values
        kv_seq_length = seq_length
        if past_key_value is not None:
            kv_seq_length += past_key_value[0].shape[1]  # Add length of cached keys
            
            # Concatenate past keys and values with current
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        
        # Create or ensure our rotary embeddings match the current sequence length
        if self.rotary_emb is None or self.rotary_emb.shape[0] < kv_seq_length:
            self.rotary_emb = precompute_rotary_embedding(
                kv_seq_length, self.head_dim, device=hidden_states.device
            )
        
        # Apply rotary positional embeddings to query and key
        query_states, key_states = apply_rotary_embeddings(
            query_states, key_states, self.rotary_emb, position_ids
        )
        
        # Handle grouped-query attention (GQA) by repeating keys and values if needed
        if self.num_key_value_heads < self.num_heads:
            # Each query head maps to a specific key/value head in a repeating pattern
            # Example: If num_heads=8, num_kv_heads=2, then head_mapping=[0,1,0,1,0,1,0,1]
            head_mapping = torch.arange(
                self.num_heads, device=hidden_states.device
            ) % self.num_key_value_heads
            
            # Expand key and value states to match the number of query heads
            # Shape: [batch_size, seq_length, num_heads, head_dim]
            key_states = torch.gather(
                key_states.repeat(1, 1, self.num_heads // self.num_key_value_heads, 1),
                dim=2,
                index=head_mapping.view(1, 1, self.num_heads, 1).expand(
                    batch_size, kv_seq_length, self.num_heads, self.head_dim
                ),
            )
            value_states = torch.gather(
                value_states.repeat(1, 1, self.num_heads // self.num_key_value_heads, 1),
                dim=2,
                index=head_mapping.view(1, 1, self.num_heads, 1).expand(
                    batch_size, kv_seq_length, self.num_heads, self.head_dim
                ),
            )
        
        # Transpose for batched matrix multiplication
        # [batch_size, num_heads, seq_length, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Calculate attention scores
        # [batch_size, num_heads, seq_length, kv_seq_length]
        attention_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        # [batch_size, num_heads, seq_length, head_dim]
        context_states = torch.matmul(attention_weights, value_states)
        
        # Transpose and reshape back
        # [batch_size, seq_length, num_heads, head_dim] -> [batch_size, seq_length, hidden_size]
        context_states = context_states.transpose(1, 2).reshape(
            batch_size, seq_length, self.hidden_size
        )
        
        # Apply output projection
        attn_output = self.o_proj(context_states)
        
        # Prepare outputs
        outputs = (attn_output,)
        
        # Add attention weights if requested
        if output_attentions:
            outputs = outputs + (attention_weights,)
            
        # Add key/value cache if requested
        if use_cache:
            past_key_value = (key_states, value_states)
            outputs = outputs + (past_key_value,)
            
        return outputs


class LlamaMLP(nn.Module):
    """
    Llama MLP (Feed-Forward Network) with SwiGLU activation.
    
    SwiGLU is a variant of GLU (Gated Linear Unit) that uses the SiLU (Swish)
    activation function instead of sigmoid, providing better performance in practice.
    
    Implemented as: SiLU(x·W1)·(x·W3)
    """
    def __init__(self, config):
        """
        Initialize the MLP module.
        
        Args:
            config: LlamaConfig instance with model configuration
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Projections for SwiGLU
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Activation function (SiLU/Swish)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        """
        Apply feed-forward network with SwiGLU activation.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            Transformed tensor of shape [batch_size, seq_length, hidden_size]
        """
        # SwiGLU: SiLU(gate_proj(x)) * up_proj(x)
        intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        
        # Project back to hidden size
        output_states = self.down_proj(intermediate_states)
        
        return output_states


class LlamaDecoderLayer(nn.Module):
    """
    Single decoder layer for Llama.
    
    Each layer consists of:
    1. RMSNorm for input
    2. Self-attention with rotary positional embeddings
    3. RMSNorm for attention output
    4. MLP with SwiGLU activation
    
    This uses a Pre-Norm architecture where normalization is applied before
    the attention and MLP blocks.
    """
    def __init__(self, config):
        """
        Initialize decoder layer.
        
        Args:
            config: LlamaConfig instance with model configuration
        """
        super().__init__()
        
        # Layer normalization before attention (Pre-Norm architecture)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Multi-head attention with rotary position embeddings
        self.self_attn = LlamaAttention(config)
        
        # Layer normalization before MLP
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MLP with SwiGLU
        self.mlp = LlamaMLP(config)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        """
        Process input through one decoder layer.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            position_ids: Optional tensor of position indices
            past_key_value: Optional tuple of cached key and value projections
            output_attentions: Whether to return attention weights
            use_cache: Whether to return key/value states for subsequent generation
            
        Returns:
            Layer output, optional attention weights, optional key/value cache
        """
        # Save residual connection
        residual = hidden_states
        
        # Normalize before attention (Pre-Norm)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        # Extract outputs
        attn_output = attn_outputs[0]
        
        # Add residual connection
        hidden_states = residual + attn_output
        
        # Save residual for next block
        residual = hidden_states
        
        # Normalize before MLP (Pre-Norm)
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP with SwiGLU
        mlp_output = self.mlp(hidden_states)
        
        # Add residual connection
        hidden_states = residual + mlp_output
        
        # Prepare outputs
        outputs = (hidden_states,)
        
        # Add attention weights if requested
        if output_attentions:
            outputs = outputs + (attn_outputs[1],)
            
        # Add key/value cache if requested
        if use_cache:
            outputs = outputs + (attn_outputs[-1],)
            
        return outputs


class LlamaModel(nn.Module):
    """
    Base Llama model consisting of embeddings and a stack of decoder layers.
    
    This is a decoder-only autoregressive transformer that can be used for various
    language generation tasks.
    """
    def __init__(self, config):
        """
        Initialize the Llama model.
        
        Args:
            config: LlamaConfig instance with model configuration
        """
        super().__init__()
        
        # Token embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Store config for reference
        self.config = config
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
    ):
        """
        Forward pass of the Llama model.
        
        Args:
            input_ids: Token ID tensor of shape [batch_size, seq_length]
            attention_mask: Optional attention mask tensor
            position_ids: Optional position ID tensor
            past_key_values: Optional list of cached key and value projections
            output_attentions: Whether to return attention weights from all layers
            output_hidden_states: Whether to return hidden states from all layers
            use_cache: Whether to return key/value states for subsequent generation
            
        Returns:
            Last hidden states, optional all hidden states, optional all attention weights,
            optional key/value cache
        """
        # Get input shape
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            # Create sequence of position indices: 0, 1, 2, ...
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
        
        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize lists for collecting outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Process through each layer
        for idx, decoder_layer in enumerate(self.layers):
            # Add to all_hidden_states if requested
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # Get past key/value for this layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None
                
            # Process through this layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            # Update hidden states
            hidden_states = layer_outputs[0]
            
            # Add to attention outputs if requested
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                
            # Add to cache if requested
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
                
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        # Add to all_hidden_states if requested
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        return tuple(v for v in [
            hidden_states,
            next_decoder_cache,
            all_hidden_states,
            all_self_attentions,
        ] if v is not None)


class LlamaForCausalLM(nn.Module):
    """
    Llama model with a language modeling head on top.
    
    This adds a linear layer on top of the base model to predict the next token.
    """
    def __init__(self, config):
        """
        Initialize Llama with language modeling head.
        
        Args:
            config: LlamaConfig instance with model configuration
        """
        super().__init__()
        
        # Base Llama model
        self.model = LlamaModel(config)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Store config for reference
        self.config = config
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
    ):
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: Token ID tensor of shape [batch_size, seq_length]
            attention_mask: Optional attention mask tensor
            position_ids: Optional position ID tensor
            past_key_values: Optional list of cached key and value projections
            labels: Optional tensor of ground truth next tokens
            output_attentions: Whether to return attention weights from all layers
            output_hidden_states: Whether to return hidden states from all layers
            use_cache: Whether to return key/value states for subsequent generation
            
        Returns:
            Loss if labels are provided, otherwise logits for next token prediction
            Optional model outputs
        """
        # Get outputs from the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        
        # Get the last hidden states
        hidden_states = outputs[0]
        
        # Apply language modeling head to predict next tokens
        logits = self.lm_head(hidden_states)
        
        # Initialize the output tuple
        output = (logits,) + outputs[1:]
        
        # If training (labels provided), compute loss
        if labels is not None:
            # Define loss function
            loss_fct = nn.CrossEntropyLoss()
            
            # Flatten the logits and labels for the loss function
            # The model predicts the next token for each position
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute loss
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            # Add loss to output
            output = (loss,) + output
            
        return output
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        """
        Prepare inputs for generation.
        
        This is a helper method used during text generation to prepare the inputs
        for the next forward pass, especially when using the key/value cache.
        
        Args:
            input_ids: Token ID tensor
            past_key_values: Optional cached key and value projections
            attention_mask: Optional attention mask tensor
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of inputs for the forward pass
        """
        # If using past key values, we only need to process the new tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
        # Get position IDs from kwargs or create default ones
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            if past_key_values is not None:
                # Get the position ID for the last token + 1
                position_ids = past_key_values[0][0].shape[1]
                position_ids = torch.tensor([position_ids], device=input_ids.device)
            else:
                # Create sequential position IDs
                position_ids = torch.arange(
                    input_ids.shape[1], dtype=torch.long, device=input_ids.device
                ).unsqueeze(0)
        
        # Return inputs as a dictionary
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }


# Utility functions for creating specific model configurations

def create_llama_7b_config():
    """
    Create a configuration for Llama 7B.
    """
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,  # Regular multi-head attention
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
    )


def create_llama_13b_config():
    """
    Create a configuration for Llama 13B.
    """
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,  # Regular multi-head attention
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
    )


def create_llama2_7b_config():
    """
    Create a configuration for Llama 2 7B.
    """
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,  # Regular multi-head attention
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
    )


def create_llama2_7b_chat_config():
    """
    Create a configuration for Llama 2 7B Chat.
    
    Same as base Llama 2 7B but specialized for dialogue.
    """
    return create_llama2_7b_config()


def create_llama2_13b_config():
    """
    Create a configuration for Llama 2 13B.
    """
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,  # Regular multi-head attention
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
    )


def create_llama2_70b_config():
    """
    Create a configuration for Llama 2 70B.
    
    This model uses Grouped-Query Attention for efficiency.
    """
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,  # Grouped-Query Attention (GQA)
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
    )


def create_llama3_8b_config():
    """
    Create a configuration for Llama 3 8B.
    """
    return LlamaConfig(
        vocab_size=128256,  # Larger vocabulary
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # Grouped-Query Attention (GQA)
        max_position_embeddings=8192,  # Longer context
        rms_norm_eps=1e-5,
    )


def create_llama3_70b_config():
    """
    Create a configuration for Llama 3 70B.
    
    This model uses Grouped-Query Attention for efficiency.
    """
    return LlamaConfig(
        vocab_size=128256,  # Larger vocabulary
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,  # Grouped-Query Attention (GQA)
        max_position_embeddings=8192,  # Longer context
        rms_norm_eps=1e-5,
    )


# Text generation functions

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.95):
    """
    Generate text using a Llama model.
    
    Args:
        model: LlamaForCausalLM instance
        tokenizer: Tokenizer for encoding/decoding text
        prompt: String prompt to start generation
        max_length: Maximum length of generated sequence (including prompt)
        temperature: Controls randomness (lower = more deterministic)
        top_p: Nucleus sampling parameter (1.0 = greedy decoding)
        
    Returns:
        Generated text as a string
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        
        # Move to the same device as the model
        input_ids = input_ids.to(next(model.parameters()).device)
        
        # Initialize generation parameters
        attention_mask = torch.ones_like(input_ids)
        past_key_values = None
        
        # Generate tokens one by one
        for _ in range(max_length - input_ids.size(1)):
            # Prepare inputs
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, past_key_values=past_key_values, attention_mask=attention_mask
            )
            
            # Forward pass
            outputs = model(**model_inputs)
            
            # Get logits for the next token
            next_token_logits = outputs[0][:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                # Convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Find the minimum number of tokens needed to exceed top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter back to original logits
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                
                # Set removed indices to negative infinity
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=1
            )
            
            # Update past_key_values
            past_key_values = outputs[1]
            
            # Stop if we generate an EOS token
            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break
                
    # Decode the generated sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return generated_text


# Testing the model

def test_llama_model():
    """
    Create and test a small Llama model.
    """
    # Create a tiny configuration for testing
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # Use GQA for testing
        max_position_embeddings=128,
    )
    
    # Create model
    model = LlamaForCausalLM(config)
    
    # Create random input data
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    
    # Get logits and past key values
    logits = outputs[0]
    past_key_values = outputs[1]
    
    # Print shapes and model info
    print("Llama model test:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Number of past key value tuples: {len(past_key_values)}")
    print(f"Shape of first past key: {past_key_values[0][0].shape}")
    print(f"Shape of first past value: {past_key_values[0][1].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    return model


if __name__ == "__main__":
    # Test the Llama model
    model = test_llama_model()
    
    # Note: To use the generate_text function, you would need a tokenizer
    # For example, using Hugging Face's transformers:
    # from transformers import LlamaTokenizer
    # tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    # generate_text(model, tokenizer, "Once upon a time")