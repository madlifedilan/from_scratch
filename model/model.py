from typing import Optional
from einops import rearrange, repeat
from transformers import PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
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
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1e6,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

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

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        return self.weight*self._norm(x.float()).type_as(x)*x
    
def precompute_freqs_cis(dim:int, end:int=32*1024, rope_base:float=1e6, rope_scaling:Optional[dict]=None):
    freqs, attn_factor = (1.0/(rope_base**(torch.arange(0, dim, 2)[:(dim//2)].float()/dim)), 1.0)

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )

        if end > orig_max:
            inv_dim = lambda b:(dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))
            low, high = (max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim//2 - 1))

            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                /max(high - low, 1e-3),
                0,
                1
            )

            freqs = freqs*(1-ramp+ramp/factor)
        
    t = torch.arange(end, device=freqs.device).float()
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)*attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)*attn_factor 

    return freqs_cos, freqs_sin
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    q, k: (b, seq, h, d)
    cos, sin: (seq, dim)
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    q_embed = rotate_half(q)
    k_embed = rotate_half(k)
    q = q * cos.unsqueeze(unsqueeze_dim) + q_embed * sin.unsqueeze(unsqueeze_dim)
    k = k * cos.unsqueeze(unsqueeze_dim) + k_embed * sin.unsqueeze(unsqueeze_dim)

    return q, k

def repeat_kv(x, num_repeats):
    if num_repeats == 1:
        return x
    
    x = repeat(x, "b seq h d -> b seq (h r) d", r=num_repeats)
    return x

class Attention(nn.Module):
    def __init__(self, args:MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads else args.num_attention_heads
        
        assert args.num_attention_heads % self.num_key_value_heads == 0,"num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads//self.num_key_value_heads
        self.head_dim = args.hidden_size//args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads*self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads*self.head_dim, args.hidden_size, bias=False)

        self.flash_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention

    def forward(self, x:torch.Tensor, position_embedding:tuple, past_key_value:Optional[tuple]=None, use_cache:bool=False, attn_mask:Optional[torch.Tensor]=None):
        """
        x: (b, seq, dim)
        position_embedding: tuple of (cos, sin) each of shape (seq, dim)
        past_key_value: tuple of (k, v) each of shape (b, num_heads, seq, head_dim)
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b seq (h d) -> b seq h d", h=self.n_local_heads)
        k = rearrange(k, "b seq (h d) -> b seq h d", h=self.num_key_value_heads)
        v = rearrange(v, "b seq (h d) -> b seq h d", h=self.num_key_value_heads)

        q, k = apply_rotary_pos_emb(q, k, position_embedding[0], position_embedding[1])

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        q = rearrange(q, "b seq h d -> b h seq d")

        k = rearrange(repeat_kv(k, self.n_rep), "b seq h d -> b h seq d")
        v = rearrange(repeat_kv(v, self.n_rep), "b seq h d -> b h seq d")

        if self.flash_attn:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(self.head_dim)
            attn_scores = attn_scores.masked_fill(attn_mask==0, float("-inf")) if attn_mask is not None else attn_scores
            attn_probs = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_probs, v)
        
        out = rearrange(out, "b h seq d -> b seq (h d)")
        out = self.o_proj(out)

        return out, past_kv
    

def SiLU(x: torch.Tensor):
    in_type = x.dtype
    x = x.to(torch.float32)
    return (x * torch.sigmoid(x)).to(in_type)

class FeedForward(nn.Module):
    def __init__(self, args:MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size*8/3)
            args.intermediate_size = 64*((intermediate_size + 63) // 64)

        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.act_fn = SiLU

    def forward(self, x:torch.Tensor):
        return self.fc2(self.act_fn(self.fc1(x))*self.gate_proj(x))