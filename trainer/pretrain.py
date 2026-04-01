"""
Language Model Pre-training Script
===================================
Main training script for the myLLM transformer model.

Features:
- Multi-GPU distributed training via DDP
- Gradient accumulation for effective larger batch sizes
- Mixed precision training (bf16/fp16)
- Smooth checkpointing and resumption
- Training progress tracking and WandB logging
- Adaptive learning rate scheduling with cosine annealing
- Automatic gradient clipping for stability

Usage:
    # Single GPU:
    python trainer/pretrain.py --epochs 1 --batch_size 32

    # Multi-GPU (automatic DDP):
    torchrun --nproc_per_node=4 trainer/pretrain.py --epochs 1 --batch_size 32

    # Resume from checkpoint:
    python trainer/pretrain.py --from_resume 1
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import myLLMConfig
from dataset.lm_dataset import PreTrainDataset
from trainer.utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode,
    setup_seed, init_model, SkipBatchSampler
)

warnings.filterwarnings('ignore')


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(epoch: int, loader, iters: int, start_step: int = 0, wandb=None) -> None:
    """
    Train for one epoch, processing all batches.
    
    Implements the training loop with:
    - Gradient accumulation for larger effective batch sizes
    - Automatic mixed precision (AMP) with gradient scaling
    - Learning rate scheduling
    - Periodic checkpoint saving
    - Training metrics logging
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing batches
        iters: Total iterations per epoch
        start_step: Starting step for resumption (default: 0)
        wandb: Optional wandb logger for tracking
    
    Global variables used:
        - model, optimizer, scaler: Training components
        - args: Command-line arguments
        - autocast_ctx: Mixed precision context
        - lm_config: Model configuration
    """
    start_time = time.time()
    
    # Iterate through batches
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # Move data to device
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        # ========== Learning Rate Schedule ==========
        # Compute learning rate for this step using cosine annealing
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== Forward Pass ==========
        with autocast_ctx:
            # Forward pass through model
            res = model(input_ids, labels=labels)
            loss = res.loss
            
            # For MoE models, add auxiliary loss (encourages load balancing)
            if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                loss = loss + res.aux_loss
            
            # Normalize loss for gradient accumulation
            loss = loss / args.accumulation_steps

        # ========== Backward Pass & Gradient Accumulation ==========
        # Scale loss and backward for mixed precision
        scaler.scale(loss).backward()

        # Update weights and optimizer step only after accumulating enough gradients
        if (step + 1) % args.accumulation_steps == 0:
            # Unscale gradients before clipping (required for mixed precision)
            scaler.unscale_(optimizer)
            
            # Clip gradients to prevent explosion during training
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Optimizer step with automatic mixed precision scaling
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next accumulation cycle
            optimizer.zero_grad(set_to_none=True)

        # ========== Logging ==========
        # Print metrics every log_interval steps or at end of epoch
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            
            # Extract loss values
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if hasattr(res, 'aux_loss') and res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            
            # Estimate remaining time (in minutes)
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # Log to console
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, '
                f'logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, '
                f'lr: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.1f}min'
            )
            
            # Log to wandb if enabled
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ========== Checkpointing ==========
        # Save checkpoint periodically or at end of epoch
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            
            # Save model weights
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # Extract raw model from DDP wrapper
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            
            # Save in fp16 for storage efficiency
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            
            # Save complete checkpoint with optimizer and scaler state
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints'
            )
            
            model.train()
            del state_dict

        # Clean up to save memory
        del input_ids, labels, res, loss


# ============================================================================
# Training Configuration & Setup
# ============================================================================

if __name__ == "__main__":
    # ========== Argument Parsing ==========
    parser = argparse.ArgumentParser(description="myLLM Pretraining")
    
    # Model output directories
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    
    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # Model architecture
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # Data and model loading
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # Monitoring
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="myLLM-Pretrain", help="wandb项目名")
    
    # Optimizations
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    
    args = parser.parse_args()

    # ========== 1. Initialize Distributed Training and Random Seeds ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. Create Directories and Initialize Configurations ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create model configuration
    lm_config = myLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    
    # Load checkpoint if resuming training
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
        if args.from_resume == 1
        else None
    )
    
    # ========== 3. Setup Mixed Precision Training ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # Enable automatic mixed precision only on GPU
    autocast_ctx = (
        nullcontext() if device_type == "cpu"
        else torch.amp.autocast(dtype=dtype)
    )
    
    # ========== 4. Initialize WandB for Experiment Tracking ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        
        # Resume wandb run if checkpoint exists
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        
        # Create descriptive run name
        wandb_run_name = (
            f"myLLM-Pretrain-Epoch-{args.epochs}-"
            f"BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        )
        
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume
        )
    
    # ========== 5. Initialize Model, Data, and Optimizer ==========
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        device=args.device
    )
    
    # Optional: Compile model for faster inference (PyTorch 2.0+)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    # Load training dataset
    train_ds = PreTrainDataset(
        args.data_path,
        tokenizer,
        max_seq_len=args.max_seq_len
    )
    
    # Setup distributed sampler if using DDP
    train_sampler = (
        DistributedSampler(train_ds)
        if dist.is_initialized()
        else None
    )
    
    # Gradient scaler for mixed precision (only needed for fp16)
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # Initialize optimizer with AdamW
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. Resume Training State from Checkpoint ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'Resumed from epoch {start_epoch}, step {start_step}')
    
    # ========== 7. Wrap Model with DDP for Distributed Training ==========
    if dist.is_initialized():
        # Ignore certain buffers that don't need synchronization
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. Main Training Loop ==========
    for epoch in range(start_epoch, args.epochs):
        # Set sampler epoch for proper shuffling in DDP
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Shuffle dataset with epoch-specific seed
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        
        # Skip steps if resuming mid-epoch
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        
        # Create batch sampler with skip functionality
        batch_sampler = SkipBatchSampler(
            train_sampler or indices,
            args.batch_size,
            skip
        )
        
        # Create data loader
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Log epoch info if skipping steps
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. Cleanup Distributed Processes ==========
    if dist.is_initialized():
        dist.destroy_process_group()
    
    Logger('Training completed successfully!')