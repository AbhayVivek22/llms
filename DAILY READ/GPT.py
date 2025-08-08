"""
GPT-2/3 ARCHITECTURE: COMPLETE FLOW VISUALIZATION

                          INPUT SEQUENCE
                               │
                               ▼
┌─────────────────────────────────────────────┐
│               TOKEN EMBEDDING               │
└───────────────────────┬─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│             POSITIONAL ENCODING             │
└───────────────────────┬─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────┐ │
│ │  DECODER BLOCK 1                        │ │
│ │                                         │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │       LAYER NORMALIZATION         │  │ │  <-- Pre-LN architecture (vs Post-LN in original Transformer)
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │    MASKED MULTI-HEAD ATTENTION    │  │ │  <-- Causal masking (each token can only see previous tokens)
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │         RESIDUAL CONNECTION       │  │ │
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │       LAYER NORMALIZATION         │  │ │
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │       FEED-FORWARD NETWORK        │  │ │
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
│              LAYER NORMALIZATION            │  <-- Final layer norm
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

GPT-2/3 KEY FEATURES:

1. Decoder-Only Architecture - No encoder, just stacked decoder blocks
2. Causal/Autoregressive Attention - Each token can only attend to itself and previous tokens
3. Pre-Layer Normalization - LayerNorm before attention and feed-forward (differs from original Transformer)
4. No Cross-Attention - Unlike the Transformer decoder, there's no encoder-decoder attention
5. Scale Differences:
   - GPT-2: Up to 1.5B parameters, 1024 context length
   - GPT-3: Up to 175B parameters, 2048 context length
6. Trained on language modeling objective (predict next token)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GPTConfig:
    """
    Configuration class to store the configuration of a GPT model.
    
    This makes it easier to create multiple variants (GPT-2, GPT-3 sizes)
    by just changing the configuration.
    """
    def __init__(
        self,
        vocab_size=50257,  # GPT-2 vocabulary size
        max_position_embeddings=1024,  # Maximum sequence length
        hidden_size=768,  # Dimension of embeddings and hidden layers
        num_hidden_layers=12,  # Number of decoder blocks
        num_attention_heads=12,  # Number of attention heads
        intermediate_size=None,  # Dimension of feed-forward network (4*hidden_size if None)
        hidden_dropout_prob=0.1,  # Dropout probability
        attention_dropout_prob=0.1,  # Dropout probability in attention
        initializer_range=0.02,  # Standard deviation for initializing weights
        layer_norm_eps=1e-5,  # Small constant for layer normalization stability
    ):
        """
        Initialize GPT configuration.
        
        Args:
            vocab_size: Size of the vocabulary
            max_position_embeddings: Maximum sequence length
            hidden_size: Dimension of embeddings and hidden layers
            num_hidden_layers: Number of decoder blocks
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of feed-forward network (4*hidden_size if None)
            hidden_dropout_prob: Dropout probability for hidden layers
            attention_dropout_prob: Dropout probability for attention probabilities
            initializer_range: Standard deviation for initializing weights
            layer_norm_eps: Small constant for layer normalization stability
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class GPTEmbeddings(nn.Module):
    """
    GPT embeddings: token embeddings + positional embeddings.
    
    Unlike BERT, GPT does not use segment/token type embeddings as it typically
    processes a single continuous text stream rather than sentence pairs.
    """
    def __init__(self, config):
        """
        Initialize embeddings module.
        
        Args:
            config: GPTConfig instance with model configuration
        """
        super().__init__()
        
        # Token embedding layer maps token IDs to hidden_size dimensional vectors
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embedding layer provides information about token position
        # GPT uses learned positional embeddings rather than the sinusoidal ones
        # from the original Transformer paper
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # Register buffer for position IDs (0, 1, 2, ...)
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Store config for reference
        self.config = config
        
    def forward(self, input_ids, position_ids=None):
        """
        Process input through the embedding layer.
        
        Args:
            input_ids: Token ID tensor of shape [batch_size, seq_length]
            position_ids: Optional position ID tensor of shape [batch_size, seq_length]
            
        Returns:
            Embedded representation of shape [batch_size, seq_length, hidden_size]
        """
        # Get sequence length from input
        seq_length = input_ids.size(1)
        
        # If position IDs are not provided, create default position IDs (0, 1, 2, ...)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        # Convert token IDs to embeddings
        token_embeddings = self.token_embeddings(input_ids)
        
        # Add position embeddings to provide position information
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings


class GPTSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism for GPT.
    
    This allows each token to attend to all previous tokens and itself,
    but not to future tokens (causal/autoregressive attention).
    """
    def __init__(self, config):
        """
        Initialize self-attention module.
        
        Args:
            config: GPTConfig instance with model configuration
        """
        super().__init__()
        
        # Ensure hidden_size is divisible by num_attention_heads
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by the number "
                f"of attention heads {config.num_attention_heads}"
            )
        
        # Save configuration parameters
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear projections for Q, K, V
        # GPT typically uses a single matrix for QKV projection (more efficient)
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        
        # Output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout for attention probabilities
        self.attn_dropout = nn.Dropout(config.attention_dropout_prob)
        
        # Dropout for output
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Register buffer for causal mask
        # This is the upper triangular part of a matrix (with 1s)
        mask = torch.tril(torch.ones((config.max_position_embeddings, 
                                      config.max_position_embeddings)))
        # Convert to binary mask with 0s in the upper triangular part (excluding diagonal)
        # Shape: [1, 1, max_position_embeddings, max_position_embeddings]
        mask = mask.view(1, 1, config.max_position_embeddings, config.max_position_embeddings)
        self.register_buffer("mask", mask)
        
    def split_heads(self, x):
        """
        Reshape tensor for multi-head attention.
        
        Args:
            x: Tensor of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            Reshaped tensor of shape [batch_size, num_attention_heads, seq_length, attention_head_size]
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # Reshape: [batch_size, seq_length, hidden_size] -> 
        # [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = x.view(*new_shape)
        
        # Transpose: [batch_size, seq_length, num_attention_heads, attention_head_size] -> 
        # [batch_size, num_attention_heads, seq_length, attention_head_size]
        return x.permute(0, 2, 1, 3)
        
    def merge_heads(self, x):
        """
        Merge attention heads back to original shape.
        
        Args:
            x: Tensor of shape [batch_size, num_attention_heads, seq_length, attention_head_size]
            
        Returns:
            Reshaped tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Transpose: [batch_size, num_attention_heads, seq_length, attention_head_size] -> 
        # [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # Reshape: [batch_size, seq_length, num_attention_heads, attention_head_size] -> 
        # [batch_size, seq_length, hidden_size]
        new_shape = x.size()[:-2] + (self.all_head_size,)
        return x.view(*new_shape)
        
    def forward(self, x, attention_mask=None):
        """
        Apply causal self-attention mechanism.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional mask tensor of shape [batch_size, 1, 1, seq_length]
                            1 indicates tokens to attend to, 0 indicates tokens to ignore
            
        Returns:
            Output tensor after self-attention
            Attention probabilities
        """
        # Get batch size and sequence length
        batch_size, seq_length = x.shape[:2]
        
        # Project input to query, key, value using a single matrix
        # Shape: [batch_size, seq_length, 3 * hidden_size]
        qkv = self.c_attn(x)
        
        # Split into query, key, value
        # Each has shape: [batch_size, seq_length, hidden_size]
        query, key, value = qkv.chunk(3, dim=2)
        
        # Reshape for multi-head attention
        # Each has shape: [batch_size, num_attention_heads, seq_length, attention_head_size]
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        # Compute attention scores
        # Shape: [batch_size, num_attention_heads, seq_length, seq_length]
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        
        # Scale attention scores by sqrt(d_k)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Create causal mask (to prevent attending to future tokens)
        # Use the pre-calculated mask, but slice it to the current sequence length
        causal_mask = self.mask[:, :, :seq_length, :seq_length]
        
        # Apply causal mask
        # This sets attention scores for future tokens to -infinity
        attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply additional attention mask if provided
        # This is used for padding tokens or prompt-tuning
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention probabilities
        attention_probs = self.attn_dropout(attention_probs)
        
        # Apply attention to value tensor
        # Shape: [batch_size, num_attention_heads, seq_length, attention_head_size]
        context = torch.matmul(attention_probs, value)
        
        # Merge attention heads
        # Shape: [batch_size, seq_length, hidden_size]
        context = self.merge_heads(context)
        
        # Apply output projection and dropout
        output = self.c_proj(context)
        output = self.resid_dropout(output)
        
        return output, attention_probs


class GPTMLP(nn.Module):
    """
    MLP (Feed-Forward Network) module used in GPT.
    
    This consists of two linear transformations with a GELU activation in between.
    """
    def __init__(self, config):
        """
        Initialize MLP module.
        
        Args:
            config: GPTConfig instance with model configuration
        """
        super().__init__()
        
        # Linear transformation from hidden_size to intermediate_size
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # Linear transformation from intermediate_size back to hidden_size
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # GELU activation function (smoother than ReLU)
        self.act = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        """
        Apply feed-forward transformations.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size]
            
        Returns:
            Transformed tensor of shape [batch_size, seq_length, hidden_size]
        """
        # First linear transformation and activation
        h = self.act(self.c_fc(x))
        
        # Second linear transformation and dropout
        h = self.c_proj(h)
        h = self.dropout(h)
        
        return h


class GPTBlock(nn.Module):
    """
    A single GPT decoder block.
    
    This consists of a multi-head self-attention layer followed by a feed-forward network,
    with layer normalization before each sub-layer (Pre-LN architecture).
    """
    def __init__(self, config):
        """
        Initialize decoder block.
        
        Args:
            config: GPTConfig instance with model configuration
        """
        super().__init__()
        
        # Layer normalization before attention (Pre-LN architecture)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Multi-head causal self-attention
        self.attn = GPTSelfAttention(config)
        
        # Layer normalization before feed-forward
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.mlp = GPTMLP(config)
        
    def forward(self, x, attention_mask=None):
        """
        Process input through one decoder block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            
        Returns:
            Processed tensor after attention and feed-forward
            Attention probabilities
        """
        # Apply layer normalization before attention (Pre-LN)
        attn_ln_output = self.ln_1(x)
        
        # Apply self-attention
        attn_output, attn_probs = self.attn(attn_ln_output, attention_mask)
        
        # Add residual connection
        # Note: Residual is added to the unnormalized input
        x = x + attn_output
        
        # Apply layer normalization before feed-forward (Pre-LN)
        mlp_ln_output = self.ln_2(x)
        
        # Apply feed-forward network
        mlp_output = self.mlp(mlp_ln_output)
        
        # Add residual connection
        x = x + mlp_output
        
        return x, attn_probs


class GPTModel(nn.Module):
    """
    The base GPT model consisting of embeddings and a stack of decoder blocks.
    
    This is a decoder-only autoregressive transformer that can be used for various
    language generation tasks.
    """
    def __init__(self, config):
        """
        Initialize the GPT model.
        
        Args:
            config: GPTConfig instance with model configuration
        """
        super().__init__()
        
        # Embedding layer (token + position)
        self.embeddings = GPTEmbeddings(config)
        
        # Stack of decoder blocks
        self.decoder_blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.num_hidden_layers)])
        
        # Final layer normalization (Post-LN for the entire model)
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Store config for reference
        self.config = config
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize the weights of the model.
        """
        # Initialize all weight matrices with normal distribution
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
                    
            # Initialize layer normalization with ones and zeros
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                
    def get_input_embeddings(self):
        """
        Get the token embedding layer.
        """
        return self.embeddings.token_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        """
        Set the token embedding layer.
        """
        self.embeddings.token_embeddings = new_embeddings
        
    def forward(self, input_ids, position_ids=None, attention_mask=None, output_attentions=False):
        """
        Forward pass of the GPT model.
        
        Args:
            input_ids: Token ID tensor of shape [batch_size, seq_length]
            position_ids: Optional position ID tensor of shape [batch_size, seq_length]
            attention_mask: Optional attention mask tensor of shape [batch_size, seq_length]
                           1 indicates tokens to attend to, 0 indicates tokens to ignore
            output_attentions: Whether to return attention weights from all layers
            
        Returns:
            Final hidden states
            All attention weights if output_attentions=True
        """
        # Process attention mask if provided
        if attention_mask is not None:
            # Convert mask of shape [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
            # Also convert mask values: 0 -> -10000.0, 1 -> 0.0
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Get embeddings for input tokens
        hidden_states = self.embeddings(input_ids, position_ids)
        
        # Store attention weights if requested
        all_attentions = [] if output_attentions else None
        
        # Process through each decoder block
        for i, block in enumerate(self.decoder_blocks):
            hidden_states, attn_weights = block(hidden_states, attention_mask)
            
            if output_attentions:
                all_attentions.append(attn_weights)
        
        # Apply final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
            
        return outputs


class GPTLMHeadModel(nn.Module):
    """
    GPT model with a language modeling head on top.
    
    This adds a linear layer on top of the base model to predict the next token.
    """
    def __init__(self, config):
        """
        Initialize GPT with language modeling head.
        
        Args:
            config: GPTConfig instance with model configuration
        """
        super().__init__()
        
        # Base GPT model
        self.transformer = GPTModel(config)
        
        # Language modeling head (tied with input embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Store config for reference
        self.config = config
        
        # Tie weights between input embeddings and output linear layer
        self._tie_weights()
        
    def _tie_weights(self):
        """
        Tie the weights between the input embeddings and the output linear layer.
        
        This is a form of weight sharing that reduces the number of parameters and
        has been shown to improve performance.
        """
        self.lm_head.weight = self.transformer.embeddings.token_embeddings.weight
        
    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None,
                output_attentions=False):
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: Token ID tensor of shape [batch_size, seq_length]
            position_ids: Optional position ID tensor of shape [batch_size, seq_length]
            attention_mask: Optional attention mask tensor of shape [batch_size, seq_length]
            labels: Optional tensor of ground truth next tokens of shape [batch_size, seq_length]
            output_attentions: Whether to return attention weights from all layers
            
        Returns:
            Loss if labels are provided, otherwise logits for next token prediction
            Model outputs from the base model
        """
        # Get outputs from the base model
        transformer_outputs = self.transformer(
            input_ids, position_ids, attention_mask, output_attentions=output_attentions
        )
        
        # Get the last hidden states
        hidden_states = transformer_outputs[0]
        
        # Apply language modeling head to predict next tokens
        lm_logits = self.lm_head(hidden_states)
        
        outputs = (lm_logits,) + transformer_outputs[1:]
        
        # If training (labels provided), compute loss
        if labels is not None:
            # Shift logits and labels for next token prediction
            # The model predicts the next token for each position
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss using cross entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            outputs = (loss,) + outputs
            
        return outputs


def create_gpt2_small_config():
    """
    Create a configuration for GPT-2 Small (124M parameters).
    """
    return GPTConfig(
        vocab_size=50257,
        max_position_embeddings=1024,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,  # 4 * hidden_size
    )


def create_gpt2_medium_config():
    """
    Create a configuration for GPT-2 Medium (355M parameters).
    """
    return GPTConfig(
        vocab_size=50257,
        max_position_embeddings=1024,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,  # 4 * hidden_size
    )


def create_gpt2_large_config():
    """
    Create a configuration for GPT-2 Large (774M parameters).
    """
    return GPTConfig(
        vocab_size=50257,
        max_position_embeddings=1024,
        hidden_size=1280,
        num_hidden_layers=36,
        num_attention_heads=20,
        intermediate_size=5120,  # 4 * hidden_size
    )


def create_gpt2_xl_config():
    """
    Create a configuration for GPT-2 XL (1.5B parameters).
    """
    return GPTConfig(
        vocab_size=50257,
        max_position_embeddings=1024,
        hidden_size=1600,
        num_hidden_layers=48,
        num_attention_heads=25,
        intermediate_size=6400,  # 4 * hidden_size
    )


def create_gpt3_small_config():
    """
    Create a configuration for a small GPT-3-like model (125M parameters).
    """
    return GPTConfig(
        vocab_size=50257,
        max_position_embeddings=2048,  # GPT-3 has longer context
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )


def create_gpt3_medium_config():
    """
    Create a configuration for a medium GPT-3-like model (350M parameters).
    """
    return GPTConfig(
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
    )


def create_gpt3_large_config():
    """
    Create a configuration for a large GPT-3-like model (6.7B parameters).
    """
    return GPTConfig(
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=16384,
    )


def create_gpt3_xl_config():
    """
    Create a configuration for an XL GPT-3-like model (13B parameters).
    
    Note: The full GPT-3 (175B) would be too large for most setups,
    so we provide a more reasonable size here.
    """
    return GPTConfig(
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=5120,
        num_hidden_layers=40,
        num_attention_heads=40,
        intermediate_size=20480,
    )


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_gpt():
    """
    Test GPT model with random input data.
    """
    # Create a small GPT-2 configuration
    config = create_gpt2_small_config()
    
    # Create model
    model = GPTLMHeadModel(config)
    
    # Create random input data
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
    
    # Get logits and attentions
    logits = outputs[0]
    attentions = outputs[1] if len(outputs) > 1 else None
    
    # Print shapes and model info
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    if attentions:
        print(f"Number of attention layers: {len(attentions)}")
        print(f"Attention shape for first layer: {attentions[0].shape}")
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:,}")
    
    return model


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.95):
    """
    Generate text using a GPT model.
    
    Args:
        model: GPTLMHeadModel instance
        tokenizer: Tokenizer for encoding/decoding text
        prompt: String prompt to start generation
        max_length: Maximum length of generated sequence (including prompt)
        temperature: Controls randomness (lower = more deterministic)
        top_k: Number of highest probability tokens to consider
        top_p: Cumulative probability threshold for nucleus sampling
        
    Returns:
        Generated text as a string
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move to the same device as the model
    input_ids = input_ids.to(next(model.parameters()).device)
    
    # Keep track of the current input sequence
    cur_input_ids = input_ids
    
    # Generate tokens one by one
    with torch.no_grad():
        for _ in range(max_length - cur_input_ids.size(1)):
            # Get predictions
            outputs = model(cur_input_ids)
            logits = outputs[0]
            
            # Get the next token logits from the last position
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                # Get top-k values and indices
                topk_values, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
                # Create a mask for non-top-k tokens
                mask = torch.zeros_like(next_token_logits).scatter_(1, topk_indices, 1)
                # Set non-top-k tokens to -inf
                next_token_logits = torch.where(mask > 0, next_token_logits, 
                                               torch.ones_like(next_token_logits) * float('-inf'))
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                # Convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Create mask for tokens within top-p cumulative probability
                mask = cumulative_probs < top_p
                # Add token with the highest probability
                mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=-1)
                # Get indices of tokens to keep
                sorted_indices_to_keep = sorted_indices.masked_select(mask)
                # Create a new mask for the original logits
                new_mask = torch.zeros_like(next_token_logits)
                new_mask.scatter_(1, sorted_indices_to_keep.unsqueeze(0), 1)
                # Apply the mask
                next_token_logits = torch.where(new_mask > 0, next_token_logits, 
                                               torch.ones_like(next_token_logits) * float('-inf'))
            
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the current sequence
            cur_input_ids = torch.cat([cur_input_ids, next_token], dim=-1)
            
            # Stop if an EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(cur_input_ids[0], skip_special_tokens=True)
    
    return generated_text


# Create a simple example of how to use a GPT model
if __name__ == "__main__":
    # Test the GPT model
    model = test_gpt()
    
    # Note: To use the generate_text function, you would need a tokenizer
    # For example, using Hugging Face's tokenizers:
    # from transformers import GPT2Tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # generate_text(model, tokenizer, "Once upon a time")