"""
myLLM: Language Model Architecture
==================================
This module defines the myLLM language model architecture, including:
- Configuration for model hyperparameters
- Core components (attention, feed-forward)
- Transformer blocks and the main model
- Inference-ready causal language model

Key Features:
- Rotary position embeddings (RoPE) with optional YaRN scaling
- Multi-head attention with key-value cache support
- Flash attention support for efficient inference
- Optional Mixture of Experts (MoE) support
- bf16/fp16 mixed precision compatibility
"""

from typing import Optional, Union, Tuple
from einops import rearrange, repeat
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math


class myLLMConfig(PretrainedConfig):
    """
    Configuration class for myLLM model.
    
    Stores all hyperparameters needed to instantiate the model,
    including embedding dimensions, layer counts, attention settings, and MoE configs.
    """
    model_type = "myllm"

    def __init__(
        self,
        # Basic model architecture
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        
        # Normalization & position embeddings
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1e6,
        inference_rope_scaling: bool = False,
        
        # Attention optimization
        flash_attention: bool = True,
        
        # Mixture of Experts (MoE) Config
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
    ):
        """Initialize configuration with all hyperparameters."""
        super().__init__(**kwargs)

        # ========== Basic Architecture ==========
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        
        # ========== Normalization & Position Embeddings ==========
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        
        # ========== Attention Optimization ==========
        self.flash_attention = flash_attention
        
        # ========== Mixture of Experts Config ==========
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        # Initialize YaRN scaling for context extension if enabled
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


# ============================================================================
# Normalization Layer: Root Mean Square Layer Normalization (RMSNorm)
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm, uses RMS (root mean square) instead of variance.
    Commonly used in modern LLMs (Llama, PaLM, etc.)
    
    Args:
        dim: Dimension of the input
        eps: Small constant for numerical stability
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """Compute RMS normalization: x / sqrt(mean(x^2) + eps)"""
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS norm: weight * (x / RMS(x))"""
        return self.weight * self._norm(x.float()).type_as(x) * x


# ============================================================================
# Position Embeddings: Rotary Position Embedding (RoPE)
# ============================================================================

def precompute_freqs_cis(
    dim: int, 
    end: int = 32 * 1024, 
    rope_base: float = 1e6, 
    rope_scaling: Optional[dict] = None
) -> Tuple[Tensor, Tensor]:
    """
    Precompute rotary position embedding frequencies (cos/sin).
    
    This precomputes the cos and sin values for Rotary Position Embeddings (RoPE),
    which is more efficient than computing them on-the-fly during forward pass.
    
    Args:
        dim: Dimension of the embedding (head_dim)
        end: Maximum sequence length
        rope_base: Base for the frequency calculation (default: 1e6)
        rope_scaling: Optional YaRN scaling config for context extension
    
    Returns:
        freqs_cos: Cosine frequencies of shape (seq_len, dim)
        freqs_sin: Sine frequencies of shape (seq_len, dim)
    """
    # Generate frequencies: 1 / (base^(2i/d)) for i in [0, d/2)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    attn_factor = 1.0

    # Apply YaRN (Yet another RoPE extensioN) scaling for context length extension
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 8192)
        factor = rope_scaling.get("factor", 8.0)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        # Calculate interpolation thresholds based on frequency bands
        inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        low, high = (
            max(math.floor(inv_dim(beta_fast)), 0), 
            min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
        )

        # Smooth interpolation between frequency bands
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 1e-3),
            0, 1
        )
        freqs = freqs * (1 - ramp + ramp / factor)
        
    # Compute the outer product of positions and frequencies
    t = torch.arange(end, device=freqs.device).float()
    freqs = torch.outer(t, freqs).float()
    
    # Duplicate for both halves of the embedding dimension and apply attention scaling
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor 

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: Tensor, 
    k: Tensor, 
    cos: Tensor, 
    sin: Tensor, 
    position_ids: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Implements the RoPE mechanism: multiplies (q, k) by a rotation matrix
    that encodes absolute position information.
    
    Args:
        q: Query tensor of shape (batch, seq, num_heads, head_dim)
        k: Key tensor of shape (batch, seq, num_heads, head_dim)
        cos: Cosine frequencies of shape (seq, dim)
        sin: Sine frequencies of shape (seq, dim)
        position_ids: Optional custom position indices (not used currently)
    
    Returns:
        q_rotated: Rotated query tensor
        k_rotated: Rotated key tensor
    """
    # Expand dimensions for broadcasting: (seq, dim) -> (1, seq, 1, dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    def rotate_half(x: Tensor) -> Tensor:
        """Rotate tensor by 90 degrees: [x1, x2] -> [-x2, x1]"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    # Apply rotation: x * cos + rotate(x) * sin
    q_embed = rotate_half(q)
    k_embed = rotate_half(k)
    q = q * cos + q_embed * sin
    k = k * cos + k_embed * sin

    return q, k


# ============================================================================
# Attention: Multi-Head Attention with KV Cache Support
# ============================================================================

def repeat_kv(x: Tensor, num_repeats: int) -> Tensor:
    """
    Repeat key/value heads for Group Query Attention (GQA) / Multi-Query Attention (MQA).
    
    In GQA, we have fewer KV heads than query heads. This function repeats
    the KV heads to match the number of query heads.
    
    Args:
        x: Tensor to repeat, shape (batch, seq, num_kv_heads, head_dim)
        num_repeats: How many times to repeat each head
    
    Returns:
        Repeated tensor of shape (batch, seq, num_query_heads, head_dim)
    """
    if num_repeats == 1:
        return x
    
    x = repeat(x, "b seq h d -> b seq (h r) d", r=num_repeats)
    return x


class Attention(nn.Module):
    """
    Multi-Head Attention with support for:
    - Flash Attention for efficient inference
    - Key-Value (KV) cache for generation
    - Group Query Attention (GQA) to reduce KV cache size
    - Causal masking for autoregressive generation
    """
    
    def __init__(self, config: myLLMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = (
            config.num_key_value_heads 
            if config.num_key_value_heads 
            else config.num_attention_heads
        )
        
        # Validate configuration
        assert (
            config.num_attention_heads % self.num_key_value_heads == 0
        ), "num_attention_heads must be divisible by num_key_value_heads"

        # Compute how many times to repeat each KV head to match Q heads
        self.n_rep = config.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Linear projections for Q, K, V, O
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # Use Flash Attention if available (CUDA-only, requires PyTorch >= 2.0)
        self.flash_attn = (
            hasattr(torch.nn.functional, 'scaled_dot_product_attention') 
            and config.flash_attention
        )

    def forward(
        self, 
        x: Tensor, 
        position_embedding: Tuple[Tensor, Tensor], 
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None, 
        use_cache: bool = False, 
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input hidden states of shape (batch, seq, hidden_size)
            position_embedding: Tuple of (cos, sin) for RoPE
            past_key_value: Optional cached key/value for generation
            use_cache: Whether to cache key/value for next step
            attn_mask: Optional attention mask
        
        Returns:
            output: Attention output of shape (batch, seq, hidden_size)
            past_kv: Optional cached (key, value) for next step
        """
        # Project and reshape for multi-head attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b seq (h d) -> b seq h d", h=self.num_attention_heads)
        k = rearrange(k, "b seq (h d) -> b seq h d", h=self.num_key_value_heads)
        v = rearrange(v, "b seq (h d) -> b seq h d", h=self.num_key_value_heads)

        # Apply rotary position embeddings
        q, k = apply_rotary_pos_emb(q, k, position_embedding[0], position_embedding[1])

        # Concatenate with past key/value if generation (KV cache)
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        
        # Prepare cache for next step
        past_kv = (k, v) if use_cache else None

        # Rearrange for efficient computation: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        q = rearrange(q, "b seq h d -> b h seq d")
        k = rearrange(repeat_kv(k, self.n_rep), "b seq h d -> b h seq d")
        v = rearrange(repeat_kv(v, self.n_rep), "b seq h d -> b h seq d")

        # Compute attention
        if self.flash_attn:
            # Flash Attention: faster and more memory-efficient
            is_causal = past_key_value and q.size(2) > 1
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                dropout_p=0.0, 
                is_causal=is_causal
            )
        else:
            # Standard attention: (Q @ K^T) / sqrt(d) @ V
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_scores = (
                attn_scores.masked_fill(attn_mask == 0, float("-inf")) 
                if attn_mask is not None 
                else attn_scores
            )
            attn_probs = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_probs, v)
        
        # Project output
        out = rearrange(out, "b h seq d -> b seq (h d)")
        out = self.o_proj(out)

        return out, past_kv
    


# ============================================================================
# Feed Forward: MLP with SiLU Activation
# ============================================================================

def SiLU(x: Tensor) -> Tensor:
    """
    Sigmoid Linear Unit activation function.
    
    Smoothly smooth activation that works well in transformers.
    Implementation: x * sigmoid(x)
    """
    in_type = x.dtype
    x = x.to(torch.float32)
    return (x * torch.sigmoid(x)).to(in_type)


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) component of transformer block.
    
    Implements the position-wise feed-forward network:
    FFN(x) = down_proj(silu(up_proj(x)) * gate_proj(x))
    
    This is the MLP part of each transformer layer, typically with
    intermediate dimension ~4x the hidden dimension.
    """
    
    def __init__(self, config: myLLMConfig):
        super().__init__()
        
        # Calculate intermediate dimension if not specified
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # Align to 64 for better GPU efficiency
            intermediate_size = 64 * ((intermediate_size + 63) // 64)
        else:
            intermediate_size = config.intermediate_size

        # Up-projection: hidden_size -> intermediate_size
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        
        # Down-projection: intermediate_size -> hidden_size
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        
        # Gate-projection: hidden_size -> intermediate_size (gating mechanism)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        
        self.act_fn = SiLU

    def forward(self, x: Tensor) -> Tensor:
        """
        Feed-forward forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq, hidden_size)
        
        Returns:
            Output tensor of shape (batch, seq, hidden_size)
        """
        return self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x))


# ============================================================================
# Transformer Block: Attention + FFN with Residual Connections
# ============================================================================

class Block(nn.Module):
    """
    Single transformer block, consisting of:
    1. Attention (with pre-normalization)
    2. Feed-Forward Network (with pre-normalization)
    3. Residual connections
    
    Pre-normalization (LayerNorm before attention/FFN) is used rather
    than post-normalization for better training stability.
    """
    
    def __init__(self, layer_id: int, config: myLLMConfig):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_id = layer_id

    def forward(
        self, 
        x: Tensor, 
        position_embedding: Tuple[Tensor, Tensor], 
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None, 
        use_cache: bool = False, 
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq, hidden_size)
            position_embedding: Rotary position embeddings
            past_key_value: Optional KV cache for generation
            use_cache: Whether to cache KV
            attn_mask: Optional attention mask
        
        Returns:
            output: Processed tensor
            past_kv: Optional cached KV for next step
        """
        # Attention with residual connection
        attn_out, past_kv = self.attn(
            self.norm1(x), 
            position_embedding, 
            past_key_value, 
            use_cache, 
            attn_mask
        )
        x = x + attn_out
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x, past_kv
    

# ============================================================================
# Transformer Model: Stack of Blocks with Embeddings
# ============================================================================

class myLLMModel(nn.Module):
    """
    Core myLLM transformer model.
    
    Architecture:
    1. Token embeddings
    2. Stack of transformer blocks with attention and FFN
    3. Final RMSNorm
    4. Precomputed rotary position embeddings (RoPE)
    """
    
    def __init__(self, config: myLLMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings: vocab_size -> hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([Block(i, config) for i in range(config.num_hidden_layers)])
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Precompute and register RoPE frequencies (no gradients needed)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            end=config.max_position_embeddings, 
            rope_base=config.rope_theta, 
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self, 
        input_ids: Tensor, 
        position_ids: Optional[Tensor] = None, 
        past_key_values: Optional[list] = None, 
        use_cache: bool = False, 
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[list]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch, seq)
            position_ids: Custom position IDs (not used currently)
            past_key_values: List of past KV caches for generation
            use_cache: Whether to cache KV for next step
            attn_mask: Optional attention mask
        
        Returns:
            hidden_states: Final hidden states of shape (batch, seq, hidden_size)
            presents: Optional list of cached KV for next steps
        """
        batch, seq = input_ids.shape
        device = input_ids.device   

        # Initialize or use provided KV cache
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.layers)

        # Calculate position embedding indices for efficiency
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        position_embedding = (
            (self.freqs_cos[start_pos:start_pos + seq], self.freqs_sin[start_pos:start_pos + seq]) 
            if self.freqs_cos is not None 
            else None
        )

        # Embed input tokens
        hidden_states = self.embed_tokens(input_ids)

        # Pass through each transformer layer
        presents = []
        for idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states, 
                position_embedding, 
                past_key_value=past_key_value, 
                use_cache=use_cache, 
                attn_mask=attn_mask
            )
            if use_cache:
                presents.append(present)

        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, presents
    

# ============================================================================
# Causal Language Model: For Pretraining and Generation
# ============================================================================

class myLLMForCausalLM(PreTrainedModel, GenerationMixin):
    """
    myLLM model with language modeling head for causal language modeling.
    
    Supports:
    - Pretraining with standard LM loss (next token prediction)
    - Generation using Auto/beam search (via GenerationMixin)
    - Efficient inference with KV cache
    - Compatibility with HuggingFace ecosystem
    """
    
    config_class = myLLMConfig
    
    def __init__(self, config: myLLMConfig):
        self.config = config
        super().__init__(config)
        
        # Core transformer model
        self.model = myLLMModel(config)
        
        # Language modeling head: hidden_size -> vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Share weights between token embeddings and output projection (for efficiency)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self, 
        input_ids: Tensor, 
        past_key_values: Optional[list] = None, 
        use_cache: bool = False, 
        attn_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        logits_to_keep: Union[int, Tensor] = 0
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Token IDs of shape (batch, seq)
            past_key_values: Optional KV cache for generation
            use_cache: Whether to cache KV
            attn_mask: Optional attention mask
            labels: Optional target token IDs for computing loss
            logits_to_keep: Keep only last N logits for efficiency (0 = keep all)
        
        Returns:
            CausalLMOutputWithPast:
                logits: Predicted logits
                loss: Optional language modeling loss (if labels provided)
                past_key_values: Optional cached KV
                hidden_states: Final hidden states
        """
        # Get hidden states from model
        hidden_states, past_key_values = self.model(
            input_ids, 
            past_key_values=past_key_values, 
            use_cache=use_cache, 
            attn_mask=attn_mask
        )
        
        # Get logits for next token prediction
        keep_num = logits_to_keep.item() if isinstance(logits_to_keep, Tensor) else logits_to_keep
        slice_indices = slice(-keep_num, None) if keep_num > 0 else slice(None)
        slice_logits = self.lm_head(hidden_states[:, slice_indices])

        return CausalLMOutputWithPast(
            logits=slice_logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )