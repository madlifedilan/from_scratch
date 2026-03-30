"""
Training Utilities
==================
This module provides helper functions and utilities for training language models.

Key Components:
1. Model utilities: Parameter counting, model initialization
2. Distributed training: DDP setup, process synchronization
3. Learning rate scheduling: Cosine annealing with warmup
4. Checkpoint management: Saving/loading model states
5. Data sampling: Batch sampling with skip functionality
6. Logging utilities: Progress tracking across processes
"""

import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model import myLLMForCausalLM
from typing import Optional, Tuple, Dict, Any


# ============================================================================
# Model Utilities
# ============================================================================

def get_model_params(model: torch.nn.Module, config) -> None:
    """
    Print detailed model parameter statistics.
    
    For standard models: prints total parameters
    For MoE models: prints total, active, base, and expert parameters separately
    
    Args:
        model: The PyTorch model to analyze
        config: Model configuration object containing MoE settings
    """
    # Calculate total parameters (in millions)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Extract MoE configuration if available
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    
    # For MoE models, calculate parameters for each expert type
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    
    # Compute base params (shared params + non-expert params)
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    
    # Compute active params (only active experts at inference time)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    
    # Log parameters
    if active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process() -> bool:
    """
    Check if current process is the main process in distributed training.
    
    Returns:
        True if not in DDP mode or rank == 0, False otherwise
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content: str) -> None:
    """
    Print content only on the main process to avoid duplicate logs.
    
    Args:
        content: Message to print
    """
    if is_main_process():
        print(content)


# ============================================================================
# Learning Rate Scheduling
# ============================================================================

def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """
    Compute learning rate with cosine annealing scheduler.
    
    This implements a cosine annealing schedule which:
    - Starts at base_lr * 1.0
    - Gradually decreases to base_lr * 0.1
    - Smooth mathematical function prevents sharp changes
    
    Formula: lr = base_lr * (0.1 + 0.45 * (1 + cos(π * step / total_steps)))
    
    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        lr: Base learning rate
    
    Returns:
        Adjusted learning rate for the current step
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# ============================================================================
# Distributed Training
# ============================================================================

def init_distributed_mode() -> int:
    """
    Initialize distributed data parallel (DDP) training.
    
    Automatically detects DDP environment and initializes process groups.
    Sets up CUDA device for the current process.
    
    Environment variables (set by torch.distributed.launch):
    - RANK: Global rank of current process
    - LOCAL_RANK: Local rank on current machine
    - MASTER_ADDR: Address of master node
    - MASTER_PORT: Port for communication
    
    Returns:
        Local CUDA device index (0-indexed). Returns 0 if not in DDP mode.
    """
    # Check if we're in distributed mode
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # Not in DDP mode
    
    # Initialize DDP with NCCL (NVIDIA Collective Communications Library)
    dist.init_process_group(backend="nccl")
    
    # Get local rank and set CUDA device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    return local_rank


def setup_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Configures:
    - Python random module
    - NumPy random generation
    - PyTorch CPU
    - PyTorch CUDA (all devices)
    - CUDNN determinism
    
    Args:
        seed: Seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Checkpointing
# ============================================================================

def lm_checkpoint(
    lm_config, 
    weight: str = 'full_sft', 
    model: Optional[torch.nn.Module] = None, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: int = 0, 
    step: int = 0, 
    wandb: Optional[Any] = None, 
    save_dir: str = '../checkpoints', 
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Save or load model checkpoint and training state.
    
    Modes:
    1. Save mode (model is not None): Saves both model weights and training state
    2. Load mode (model is None): Loads checkpoint if it exists, handles world size changes
    
    Files created:
    - {weight}_{hidden_size}.pth: Model weights only (in fp16)
    - {weight}_{hidden_size}_resume.pth: Full training state for resumption
    
    Args:
        lm_config: Model configuration object
        weight: Weight name prefix (default: 'full_sft')
        model: Model to save (None to load)
        optimizer: Optimizer state to save
        scaler: GradScaler state to save (for mixed precision)
        epoch: Current training epoch
        step: Current training step
        wandb: Optional wandb run object for logging
        save_dir: Directory to save checkpoints
        **kwargs: Additional objects with state_dict() to save
    
    Returns:
        Loaded checkpoint dict if loading mode, None otherwise
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct file paths based on configuration
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ========== SAVE MODE ==========
        
        # Extract raw model from DDP wrapper if needed
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)  # Handle torch.compile
        
        # Get state dict and convert to fp16 for storage efficiency
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        
        # Save model weights with atomic write (write to temp file first)
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        # Extract wandb run ID if logging
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # Build complete training state for resumption
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        
        # Add any additional state dicts from kwargs
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # Save with atomic write
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        
        # Clean up memory
        del state_dict, resume_data
        torch.cuda.empty_cache()
        
    else:
        # ========== LOAD MODE ==========
        
        if os.path.exists(resume_path):
            # Load checkpoint from disk
            ckp_data = torch.load(resume_path, map_location='cpu')
            
            # Handle world size changes (e.g., training on 2 GPUs, resuming on 4 GPUs)
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            
            if saved_ws != current_ws:
                # Adjust step count proportionally to world size
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            
            return ckp_data
        
        return None


# ============================================================================
# Model Initialization
# ============================================================================

def init_model(
    lm_config, 
    from_weight: str = 'pretrain', 
    tokenizer_path: str = '../model', 
    save_dir: str = '../out', 
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, Any]:
    """
    Initialize a language model with pre-trained weights if specified.
    
    Args:
        lm_config: Model configuration
        from_weight: Which pre-trained weights to load ('none' for random init)
        tokenizer_path: Path to pre-trained tokenizer
        save_dir: Directory where saved weights are stored
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create model with random initialization
    model = myLLMForCausalLM(lm_config)

    # Load pre-trained weights if specified
    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        
        # Load weights with strict=False to gracefully handle shape mismatches
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        Logger(f'Loaded weights from {weight_path}')

    # Print model statistics
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    
    return model.to(device), tokenizer


# ============================================================================
# Data Sampling
# ============================================================================

class SkipBatchSampler(Sampler):
    """
    Batch sampler that skips the first N batches (for training resumption).
    
    Useful for resuming training: after loading a checkpoint, we skip the batches
    that were already processed to avoid training redundantly.
    
    Example:
        sampler = SkipBatchSampler(indices, batch_size=32, skip_batches=10)
        # First 10 batches will be skipped
    """
    
    def __init__(self, sampler: Sampler, batch_size: int, skip_batches: int = 0):
        """
        Args:
            sampler: Base sampler providing indices
            batch_size: Size of each batch
            skip_batches: Number of batches to skip at the beginning
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """Iterate over batches, skipping the first N as configured."""
        batch = []
        skipped = 0
        
        for idx in self.sampler:
            batch.append(idx)
            
            # When batch is full, check if we should skip it
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    # Skip this batch
                    skipped += 1
                    batch = []
                    continue
                # Yield complete batch
                yield batch
                batch = []
        
        # Yield remaining samples if any (and only if we've skipped enough batches)
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self) -> int:
        """Return total number of batches after skipping."""
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)