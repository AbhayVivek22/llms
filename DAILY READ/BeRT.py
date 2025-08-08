"""
BERT/RoBERTa ARCHITECTURE: COMPLETE FLOW VISUALIZATION

                          INPUT SEQUENCE
                               │
                               ▼
┌─────────────────────────────────────────────┐
│               TOKEN EMBEDDING               │
└───────────────────────┬─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│             SEGMENT EMBEDDING               │ <-- Distinguishes sentence pairs (BERT only)
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
│ │  ENCODER LAYER 1                        │ │
│ │                                         │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │      MULTI-HEAD SELF-ATTENTION    │  │ │
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │       ADD & NORM (RESIDUAL)       │  │ │
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │       FEED-FORWARD NETWORK        │  │ │
│ │  └─────────────────┬─────────────────┘  │ │
│ │                    │                    │ │
│ │                    ▼                    │ │
│ │  ┌───────────────────────────────────┐  │ │
│ │  │       ADD & NORM (RESIDUAL)       │  │ │
│ │  └─────────────────┬─────────────────┘  │ │
│ └────────────────────┼─────────────────────┘
│                      │
│                      ▼
│ ┌─────────────────────────────────────────┐
│ │  ENCODER LAYER 2...N                    │
│ └────────────────────┼─────────────────────┘
└──────────────────────┼──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              CONTEXTUAL EMBEDDINGS          │
└───────────────────────┬─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│ TASK-SPECIFIC OUTPUTS (MLM, Classification) │
└─────────────────────────────────────────────┘

BERT/RoBERTa KEY FEATURES:

1. Bidirectional Attention - Each token attends to all other tokens
2. Pre-training Tasks:
   - Masked Language Modeling (MLM): Predict masked tokens
   - Next Sentence Prediction (NSP): Predict if sentences are consecutive (BERT only)
3. Special Tokens:
   - [CLS] at the beginning for classification tasks
   - [SEP] to separate sentences or mark the end
4. No decoder - Encoder-only architecture
5. RoBERTa improvements:
   - No NSP task
   - Dynamic masking
   - Larger batches
   - Byte-level BPE tokenizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class BertEmbedding(nn.Module):
    """
    BERT embedding layer that combines token, position, and segment embeddings.
    
    This is the first layer of BERT that converts token IDs to vector representations
    and adds information about token position and which segment (sentence) they belong to.
    """
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, 
                 type_vocab_size=2, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize BERT embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Dimension of the embeddings
            max_position_embeddings: Maximum sequence length supported
            type_vocab_size: Number of segment types (usually 2: sentence A and B)
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Token embedding layer maps token IDs to hidden_size dimensional vectors
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Position embedding layer provides information about token position
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Segment/token type embedding distinguishes between different sentences
        # (e.g., for sentence pair tasks like NSP in BERT)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        # Layer normalization for stabilizing learning
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Register buffer for position IDs
        self.register_buffer(
            "position_ids", 
            torch.arange(max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """
        Process input through the embedding layer.
        
        Args:
            input_ids: Token ID tensor of shape [batch_size, seq_length]
            token_type_ids: Optional segment ID tensor of shape [batch_size, seq_length]
            position_ids: Optional position ID tensor of shape [batch_size, seq_length]
            
        Returns:
            Embedded representation of shape [batch_size, seq_length, hidden_size]
        """
        seq_length = input_ids.size(1)
        
        # If position IDs are not provided, create default position IDs (0, 1, 2, ...)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        # If token_type_ids are not provided, create zeros (all tokens belong to first segment)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Convert token IDs to embeddings
        word_embeddings = self.word_embeddings(input_ids)
        
        # Add position embeddings to provide position information
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add segment embeddings to distinguish between sentences
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine all three embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for BERT.
    
    This allows tokens to attend to all other tokens in the sequence, capturing
    contextual relationships in a bidirectional manner.
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        """
        Initialize the self-attention module.
        
        Args:
            hidden_size: Dimension of input and output
            num_attention_heads: Number of attention heads
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        # Ensure hidden_size is divisible by num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) must be divisible by the number of attention heads ({num_attention_heads})"
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear layers for creating query, key, and value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Dropout for attention probabilities
        self.dropout = nn.Dropout(dropout_prob)
        
    def transpose_for_scores(self, x):
        """
        Reshape tensor for multi-head attention.
        
        Args:
            x: Tensor of shape [batch_size, seq_length, all_head_size]
            
        Returns:
            Reshaped tensor of shape [batch_size, num_attention_heads, seq_length, attention_head_size]
        """
        # Reshape to separate the head dimension
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        
        # Transpose to get [batch_size, num_attention_heads, seq_length, attention_head_size]
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Apply self-attention mechanism.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional mask tensor of shape [batch_size, 1, 1, seq_length]
                            1 indicates tokens to attend to, 0 indicates tokens to ignore
            
        Returns:
            Context layer tensor of shape [batch_size, seq_length, hidden_size]
            Attention probabilities tensor of shape [batch_size, num_attention_heads, seq_length, seq_length]
        """
        # Create query, key, and value projections
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate attention scores
        # (batch_size, num_heads, seq_length, seq_length)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Add a large negative value to masked positions to make their softmax -> 0
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # Apply dropout to attention probabilities
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Transpose and reshape back
        # (batch_size, seq_length, num_attention_heads, attention_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Combine heads
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    """
    Processes the output of the self-attention layer with a linear transformation,
    followed by dropout, residual connection, and layer normalization.
    """
    def __init__(self, hidden_size, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize the self-output processing module.
        
        Args:
            hidden_size: Dimension of input and output
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Linear projection to maintain the hidden_size dimension
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization for stabilizing learning
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        """
        Process the self-attention output.
        
        Args:
            hidden_states: Output tensor from self-attention
            input_tensor: Original input to the self-attention (for residual connection)
            
        Returns:
            Processed tensor after projection, dropout, residual connection, and normalization
        """
        # Apply linear projection
        hidden_states = self.dense(hidden_states)
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Add residual connection and apply layer normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states


class BertAttention(nn.Module):
    """
    Combines the self-attention mechanism with output processing.
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize the complete attention module.
        
        Args:
            hidden_size: Dimension of input and output
            num_attention_heads: Number of attention heads
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Self-attention layer
        self.self = BertSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        
        # Output processing layer
        self.output = BertSelfOutput(hidden_size, dropout_prob, layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Apply attention mechanism and process the output.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Processed attention output
            Attention probabilities
        """
        # Apply self-attention
        self_output, attention_probs = self.self(hidden_states, attention_mask)
        
        # Process the self-attention output with residual connection
        attention_output = self.output(self_output, hidden_states)
        
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    """
    Feed-forward network that processes attention output.
    
    This corresponds to the position-wise feed-forward network in the original
    Transformer paper, with GELU activation instead of ReLU.
    """
    def __init__(self, hidden_size, intermediate_size):
        """
        Initialize the feed-forward network.
        
        Args:
            hidden_size: Dimension of input
            intermediate_size: Dimension of hidden layer (usually 4x hidden_size)
        """
        super().__init__()
        
        # Linear transformation to intermediate size
        self.dense = nn.Linear(hidden_size, intermediate_size)
        
        # GELU activation function (smoother than ReLU)
        self.intermediate_act_fn = nn.GELU()
        
    def forward(self, hidden_states):
        """
        Apply feed-forward transformation with GELU activation.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Transformed tensor after feed-forward network
        """
        # Apply linear transformation
        hidden_states = self.dense(hidden_states)
        
        # Apply GELU activation
        hidden_states = self.intermediate_act_fn(hidden_states)
        
        return hidden_states


class BertOutput(nn.Module):
    """
    Processes the output of the feed-forward network with a linear transformation,
    followed by dropout, residual connection, and layer normalization.
    """
    def __init__(self, hidden_size, intermediate_size, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize the output processing module.
        
        Args:
            hidden_size: Dimension of output
            intermediate_size: Dimension of input (from feed-forward network)
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Linear transformation from intermediate_size back to hidden_size
        self.dense = nn.Linear(intermediate_size, hidden_size)
        
        # Layer normalization for stabilizing learning
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        """
        Process the feed-forward network output.
        
        Args:
            hidden_states: Output tensor from feed-forward network
            input_tensor: Original input to the feed-forward (for residual connection)
            
        Returns:
            Processed tensor after projection, dropout, residual connection, and normalization
        """
        # Apply linear projection
        hidden_states = self.dense(hidden_states)
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Add residual connection and apply layer normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states


class BertLayer(nn.Module):
    """
    A single transformer layer for BERT, consisting of multi-head self-attention
    followed by a feed-forward network, with residual connections and layer normalization.
    """
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, 
                 dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize a complete BERT layer.
        
        Args:
            hidden_size: Dimension of input and output
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of feed-forward network
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Multi-head attention with output processing
        self.attention = BertAttention(
            hidden_size, num_attention_heads, dropout_prob, layer_norm_eps
        )
        
        # Feed-forward network
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        
        # Output processing for feed-forward network
        self.output = BertOutput(
            hidden_size, intermediate_size, dropout_prob, layer_norm_eps
        )
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Process input through one BERT layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Processed tensor after attention and feed-forward
            Attention probabilities
        """
        # Apply attention and get output
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        
        # Apply feed-forward network to attention output
        intermediate_output = self.intermediate(attention_output)
        
        # Process feed-forward output with residual connection to attention output
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output, attention_probs


class BertEncoder(nn.Module):
    """
    Stack of transformer layers that forms the BERT encoder.
    """
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, 
                 num_hidden_layers, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize the BERT encoder with multiple layers.
        
        Args:
            hidden_size: Dimension of input and output
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of feed-forward network
            num_hidden_layers: Number of transformer layers
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Create a stack of identical BERT layers
        self.layers = nn.ModuleList([
            BertLayer(
                hidden_size, num_attention_heads, intermediate_size, 
                dropout_prob, layer_norm_eps
            )
            for _ in range(num_hidden_layers)
        ])
        
    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=False):
        """
        Process input through the entire encoder stack.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_all_encoded_layers: Whether to return outputs of all layers or just the last
            
        Returns:
            All layer outputs if output_all_encoded_layers=True, otherwise just the last layer
            Dictionary of attention probabilities for each layer
        """
        all_encoder_layers = []
        all_attention_probs = {}
        
        # Process through each layer sequentially
        for i, layer in enumerate(self.layers):
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                
            all_attention_probs[f'layer_{i}'] = attention_probs
        
        # If we don't need all layers, just add the last one
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        return all_encoder_layers, all_attention_probs


class BertPooler(nn.Module):
    """
    Pools the output of BERT by taking the hidden state of the first token ([CLS])
    and applying a linear layer followed by tanh activation.
    
    This is used for classification tasks where a single vector representation is needed.
    """
    def __init__(self, hidden_size):
        """
        Initialize the pooler.
        
        Args:
            hidden_size: Dimension of input and output
        """
        super().__init__()
        
        # Linear transformation for the [CLS] token representation
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        # Tanh activation function
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        """
        Pool the output to get a single vector for classification.
        
        Args:
            hidden_states: Output tensor from BERT encoder, shape [batch_size, seq_length, hidden_size]
            
        Returns:
            Pooled output for classification, shape [batch_size, hidden_size]
        """
        # Take the representation of the [CLS] token (first token)
        # Shape: [batch_size, hidden_size]
        first_token_tensor = hidden_states[:, 0]
        
        # Apply linear transformation and tanh activation
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """
    Transformation applied before the LM prediction head.
    """
    def __init__(self, hidden_size, layer_norm_eps=1e-12):
        """
        Initialize the transform module.
        
        Args:
            hidden_size: Dimension of input and output
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Linear transformation
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        # GELU activation
        self.transform_act_fn = nn.GELU()
        
        # Layer normalization
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
    def forward(self, hidden_states):
        """
        Apply transformation.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Transformed tensor
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """
    Prediction head for the masked language modeling (MLM) task.
    """
    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12):
        """
        Initialize the MLM prediction head.
        
        Args:
            hidden_size: Dimension of input
            vocab_size: Size of vocabulary for prediction
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Transformation before prediction
        self.transform = BertPredictionHeadTransform(hidden_size, layer_norm_eps)
        
        # Linear layer to predict vocabulary
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Bias parameter
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        
    def forward(self, hidden_states):
        """
        Predict token probabilities for MLM.
        
        Args:
            hidden_states: Input tensor from BERT encoder
            
        Returns:
            Logits for vocabulary prediction
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    """
    Complete MLM head including transformation and prediction.
    """
    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12):
        """
        Initialize the MLM head.
        
        Args:
            hidden_size: Dimension of input
            vocab_size: Size of vocabulary for prediction
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        self.predictions = BertLMPredictionHead(hidden_size, vocab_size, layer_norm_eps)
        
    def forward(self, sequence_output):
        """
        Predict masked tokens.
        
        Args:
            sequence_output: Output tensor from BERT encoder
            
        Returns:
            Prediction logits for masked tokens
        """
        return self.predictions(sequence_output)


class BertOnlyNSPHead(nn.Module):
    """
    Prediction head for the next sentence prediction (NSP) task.
    """
    def __init__(self, hidden_size):
        """
        Initialize the NSP head.
        
        Args:
            hidden_size: Dimension of input
        """
        super().__init__()
        # Linear layer for binary classification
        self.seq_relationship = nn.Linear(hidden_size, 2)
        
    def forward(self, pooled_output):
        """
        Predict whether two sentences are consecutive.
        
        Args:
            pooled_output: Pooled output from BERT (representation of [CLS] token)
            
        Returns:
            Logits for binary classification
        """
        return self.seq_relationship(pooled_output)


class BertPreTrainingHeads(nn.Module):
    """
    Combines the MLM and NSP prediction heads for pre-training.
    """
    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12):
        """
        Initialize both pre-training heads.
        
        Args:
            hidden_size: Dimension of input
            vocab_size: Size of vocabulary for MLM
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # MLM prediction head
        self.predictions = BertLMPredictionHead(hidden_size, vocab_size, layer_norm_eps)
        
        # NSP prediction head
        self.seq_relationship = nn.Linear(hidden_size, 2)
        
    def forward(self, sequence_output, pooled_output):
        """
        Perform both MLM and NSP predictions.
        
        Args:
            sequence_output: Output tensor from BERT encoder for MLM
            pooled_output: Pooled output from BERT for NSP
            
        Returns:
            Prediction logits for MLM and NSP
        """
        # MLM predictions on sequence output
        prediction_scores = self.predictions(sequence_output)
        
        # NSP prediction on pooled output
        seq_relationship_score = self.seq_relationship(pooled_output)
        
        return prediction_scores, seq_relationship_score


class BertModel(nn.Module):
    """
    The base BERT model with embeddings, encoder, and pooler.
    
    This provides contextual representations of the input and can be fine-tuned
    for various downstream tasks.
    """
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512,
                 type_vocab_size=2, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize the complete BERT model.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Dimension of the embeddings and hidden layers
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of the feed-forward network
            max_position_embeddings: Maximum sequence length supported
            type_vocab_size: Number of segment types
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Embedding layer (token + position + segment)
        self.embeddings = BertEmbedding(
            vocab_size, hidden_size, max_position_embeddings, 
            type_vocab_size, dropout_prob, layer_norm_eps
        )
        
        # Encoder with multiple transformer layers
        self.encoder = BertEncoder(
            hidden_size, num_attention_heads, intermediate_size, num_hidden_layers,
            dropout_prob, layer_norm_eps
        )
        
        # Pooler for classification tasks
        self.pooler = BertPooler(hidden_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize the weights of the model.
        """
        # Initialize all weight matrices with normal distribution
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            
            # Initialize all bias terms with zeros
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
            # Initialize layer normalization with ones and zeros
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
                output_all_encoded_layers=False):
        """
        Forward pass of the BERT model.
        
        Args:
            input_ids: Token ID tensor of shape [batch_size, seq_length]
            token_type_ids: Optional segment ID tensor of shape [batch_size, seq_length]
            attention_mask: Optional attention mask tensor of shape [batch_size, seq_length]
            output_all_encoded_layers: Whether to return all encoder layers
            
        Returns:
            All encoder layer outputs or just the last layer
            Pooled output for classification
            Dictionary of attention probabilities
        """
        # Process attention mask if provided
        if attention_mask is not None:
            # Create attention mask for self-attention
            # Convert mask of shape [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
            # 1.0 for tokens to attend to, 0.0 for tokens to ignore
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert mask values: 0 -> -10000.0, 1 -> 0.0
            # This will be added to attention scores
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Get embeddings for input tokens
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Process through encoder
        encoded_layers, all_attention_probs = self.encoder(
            embedding_output, extended_attention_mask, output_all_encoded_layers
        )
        
        # Get the final encoder layer output
        sequence_output = encoded_layers[-1]
        
        # Get pooled output for the [CLS] token
        pooled_output = self.pooler(sequence_output)
        
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
            
        return encoded_layers, pooled_output, all_attention_probs


class BertForPreTraining(nn.Module):
    """
    BERT model with pre-training heads for masked language modeling (MLM)
    and next sentence prediction (NSP).
    """
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512,
                 type_vocab_size=2, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize BERT with pre-training heads.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Dimension of the embeddings and hidden layers
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of the feed-forward network
            max_position_embeddings: Maximum sequence length supported
            type_vocab_size: Number of segment types
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Base BERT model
        self.bert = BertModel(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, max_position_embeddings, type_vocab_size,
            dropout_prob, layer_norm_eps
        )
        
        # Add pre-training heads
        self.cls = BertPreTrainingHeads(hidden_size, vocab_size, layer_norm_eps)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize the weights of the pre-training heads.
        """
        # Initialize all weight matrices with normal distribution
        for module in self.cls.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        """
        Forward pass for pre-training.
        
        Args:
            input_ids: Token ID tensor
            token_type_ids: Optional segment ID tensor
            attention_mask: Optional attention mask tensor
            masked_lm_labels: Optional tensor of labels for masked tokens
            next_sentence_label: Optional tensor of labels for NSP
            
        Returns:
            Total loss or individual task losses if labels are provided,
            otherwise prediction logits
        """
        # Get BERT outputs
        sequence_output, pooled_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        
        # Get prediction scores for MLM and NSP
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        # If training (labels provided), compute losses
        if masked_lm_labels is not None and next_sentence_label is not None:
            # Loss for MLM
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            
            # Loss for NSP
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            )
            
            # Total loss
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            # If not training, return prediction scores
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(nn.Module):
    """
    BERT model with a masked language modeling (MLM) head.
    
    This is used for the MLM pre-training task or for tasks like fill-in-the-blank.
    """
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512,
                 type_vocab_size=2, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize BERT with MLM head.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Dimension of the embeddings and hidden layers
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of the feed-forward network
            max_position_embeddings: Maximum sequence length supported
            type_vocab_size: Number of segment types
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Base BERT model
        self.bert = BertModel(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, max_position_embeddings, type_vocab_size,
            dropout_prob, layer_norm_eps
        )
        
        # MLM prediction head
        self.cls = BertOnlyMLMHead(hidden_size, vocab_size, layer_norm_eps)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize the weights of the MLM head.
        """
        for module in self.cls.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        """
        Forward pass for masked language modeling.
        
        Args:
            input_ids: Token ID tensor
            token_type_ids: Optional segment ID tensor
            attention_mask: Optional attention mask tensor
            masked_lm_labels: Optional tensor of labels for masked tokens
            
        Returns:
            MLM loss if labels are provided, otherwise prediction logits
        """
        # Get BERT outputs
        sequence_output, _, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        
        # Get prediction scores for MLM
        prediction_scores = self.cls(sequence_output)
        
        # If training (labels provided), compute loss
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            return masked_lm_loss
        else:
            return prediction_scores


class BertForSequenceClassification(nn.Module):
    """
    BERT model for sequence classification tasks like sentiment analysis.
    
    This adds a linear layer on top of the pooled output from BERT.
    """
    def __init__(self, vocab_size, num_labels, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512,
                 type_vocab_size=2, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize BERT for sequence classification.
        
        Args:
            vocab_size: Size of the vocabulary
            num_labels: Number of classification labels
            hidden_size: Dimension of the embeddings and hidden layers
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of the feed-forward network
            max_position_embeddings: Maximum sequence length supported
            type_vocab_size: Number of segment types
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Base BERT model
        self.bert = BertModel(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, max_position_embeddings, type_vocab_size,
            dropout_prob, layer_norm_eps
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize the weights of the classification head.
        """
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
            
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for sequence classification.
        
        Args:
            input_ids: Token ID tensor
            token_type_ids: Optional segment ID tensor
            attention_mask: Optional attention mask tensor
            labels: Optional tensor of class labels
            
        Returns:
            Classification loss if labels are provided, otherwise logits
        """
        # Get BERT outputs
        _, pooled_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # If training (labels provided), compute loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForTokenClassification(nn.Module):
    """
    BERT model for token classification tasks like named entity recognition.
    
    This adds a linear layer on top of the hidden states output from BERT.
    """
    def __init__(self, vocab_size, num_labels, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512,
                 type_vocab_size=2, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize BERT for token classification.
        
        Args:
            vocab_size: Size of the vocabulary
            num_labels: Number of classification labels
            hidden_size: Dimension of the embeddings and hidden layers
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of the feed-forward network
            max_position_embeddings: Maximum sequence length supported
            type_vocab_size: Number of segment types
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Base BERT model
        self.bert = BertModel(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, max_position_embeddings, type_vocab_size,
            dropout_prob, layer_norm_eps
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Classification head for each token
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize the weights of the classification head.
        """
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
            
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for token classification.
        
        Args:
            input_ids: Token ID tensor
            token_type_ids: Optional segment ID tensor
            attention_mask: Optional attention mask tensor
            labels: Optional tensor of token labels
            
        Returns:
            Classification loss if labels are provided, otherwise logits
        """
        # Get BERT outputs
        sequence_output, _, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        
        # Apply dropout and classification head
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # If training (labels provided), compute loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits


class BertForQuestionAnswering(nn.Module):
    """
    BERT model for question answering tasks like SQuAD.
    
    This adds a linear layer to predict the start and end positions of the answer span.
    """
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512,
                 type_vocab_size=2, dropout_prob=0.1, layer_norm_eps=1e-12):
        """
        Initialize BERT for question answering.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Dimension of the embeddings and hidden layers
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of the feed-forward network
            max_position_embeddings: Maximum sequence length supported
            type_vocab_size: Number of segment types
            dropout_prob: Dropout probability
            layer_norm_eps: Small constant for layer normalization
        """
        super().__init__()
        
        # Base BERT model
        self.bert = BertModel(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, max_position_embeddings, type_vocab_size,
            dropout_prob, layer_norm_eps
        )
        
        # QA output layer to predict start and end positions
        self.qa_outputs = nn.Linear(hidden_size, 2)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize the weights of the QA output layer.
        """
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        if self.qa_outputs.bias is not None:
            self.qa_outputs.bias.data.zero_()
            
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
                start_positions=None, end_positions=None):
        """
        Forward pass for question answering.
        
        Args:
            input_ids: Token ID tensor
            token_type_ids: Optional segment ID tensor
            attention_mask: Optional attention mask tensor
            start_positions: Optional tensor of ground truth start positions
            end_positions: Optional tensor of ground truth end positions
            
        Returns:
            Total loss if start and end positions are provided, otherwise start and end logits
        """
        # Get BERT outputs
        sequence_output, _, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        
        # Predict start and end positions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # If training (positions provided), compute loss
        if start_positions is not None and end_positions is not None:
            # Sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


# Example of creating and using the BERT model

def create_sample_bert():
    """
    Create a sample BERT model for demonstration.
    """
    # Configuration similar to BERT-Base
    vocab_size = 30522  # Standard vocabulary size for BERT
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    
    # Create the model
    model = BertModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size
    )
    
    return model


def create_sample_roberta():
    """
    Create a sample RoBERTa model for demonstration.
    
    RoBERTa has the same architecture as BERT but with different pre-training
    and the following differences:
    - No NSP task
    - Larger batch size
    - Dynamic masking (new masks each time a sequence is seen)
    - Byte-level BPE tokenizer
    """
    # Configuration similar to RoBERTa-Base
    vocab_size = 50265  # RoBERTa uses a larger vocabulary
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    
    # Create the model (same architecture as BERT)
    model = BertModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size
    )
    
    return model


def test_bert():
    """
    Test a BERT model with random input data.
    """
    # Create sample model
    model = create_sample_bert()
    
    # Create random input data
    batch_size = 2
    seq_length = 10
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    encoded_layers, pooled_output, attention_probs = model(
        input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True
    )
    
    # Print shapes
    print(f"Input shape: {input_ids.shape}")
    print(f"Number of encoder layers: {len(encoded_layers)}")
    print(f"Last encoder layer shape: {encoded_layers[-1].shape}")
    print(f"Pooled output shape: {pooled_output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model


if __name__ == "__main__":
    # Test the BERT model
    model = test_bert()