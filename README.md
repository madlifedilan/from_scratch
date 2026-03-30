# myLLM: Language Model from Scratch

A clean, well-documented implementation of a transformer-based language model with support for distributed training, mixed precision, and modern optimization techniques.

## Project Structure

```
project/
├── model/
│   └── model.py              # Transformer architecture with RoPE and attention
├── dataset/
│   └── lm_dataset.py         # Data loading and preprocessing
├── trainer/
│   ├── pretrain.py           # Main training script
│   └── utils.py              # Training utilities and helpers
└── README.md                 # This file
```

## Module Overview

### 📋 model/model.py - Transformer Architecture

**Key Components:**

- **myLLMConfig**: Configuration class storing all hyperparameters
- **RMSNorm**: Root mean square layer normalization (more efficient than LayerNorm)
- **Rotary Position Embeddings (RoPE)**: Context-aware position encoding with optional YaRN scaling
- **Attention**: Multi-head attention with:
  - Flash Attention support for efficient inference
  - Key-value cache for autoregressive generation
  - Group Query Attention (GQA) for reduced KV cache size
- **FeedForward**: Position-wise FFN with gating mechanism
- **Block**: Transformer block combining attention and FFN with residuals
- **myLLMModel**: Stack of transformer blocks
- **myLLMForCausalLM**: Language model wrapper for training and generation

**Features:**
- Mixed precision (bf16/fp16) compatibility
- Efficient KV caching for streaming generation
- Optional Mixture of Experts (MoE) support
- Weight sharing between embeddings and output layer

### 📊 dataset/lm_dataset.py - Data Processing

**PreTrainDataset Class:**

Handles efficient loading and preprocessing of JSONL dataset files.

**Processing Pipeline:**
1. Load text from JSONL files using HuggingFace datasets
2. Tokenize with automatic truncation
3. Add special tokens: [BOS] + tokens + [EOS]
4. Pad sequences to max_seq_len
5. Mask padding tokens in labels (-100 for loss computation)

**Features:**
- Lazy loading for memory efficiency with large datasets
- Automatic handling of special tokens
- Proper padding and sequence masking

### 🔧 trainer/utils.py - Training Utilities

**Key Functions:**

1. **Model Utilities:**
   - `get_model_params()`: Print detailed parameter statistics
   - `init_model()`: Initialize model with optional pre-trained weights
   - `is_main_process()`: Check if current process is main in DDP

2. **Learning Rate Scheduling:**
   - `get_lr()`: Cosine annealing with warmup

3. **Distributed Training:**
   - `init_distributed_mode()`: Setup DDP with automatic CUDA device assignment
   - `setup_seed()`: Reproducible random seeding across all libraries

4. **Checkpointing:**
   - `lm_checkpoint()`: Save/load model weights and training state
   - Handles world-size changes gracefully for multi-GPU resumption

5. **Data Sampling:**
   - `SkipBatchSampler`: Skip batches for mid-epoch training resumption

6. **Logging:**
   - `Logger()`: Print only on main process to avoid duplicate logs

### 🚀 trainer/pretrain.py - Training Script

**Main Training Loop:**

Implements complete training pipeline with:

**Features:**
- ✅ Multi-GPU distributed training (DDP)
- ✅ Gradient accumulation for effective larger batch sizes
- ✅ Mixed precision training (bf16/fp16)
- ✅ Gradient clipping for training stability
- ✅ Periodic checkpointing and resumption
- ✅ Learning rate scheduling with cosine annealing
- ✅ Training metrics logging and WandB integration

**Key Functions:**

- `train_epoch()`: Main training loop for a single epoch
  - Forward pass with mixed precision
  - Gradient accumulation and backprop
  - Learning rate scheduling per step
  - Periodic checkpoint saving
  - Training metrics logging

**Usage:**

```bash
# Single GPU training
python trainer/pretrain.py --epochs 1 --batch_size 32

# Multi-GPU distributed training (automatic DDP)
torchrun --nproc_per_node=4 trainer/pretrain.py --epochs 1 --batch_size 32

# Resume from checkpoint
python trainer/pretrain.py --from_resume 1

# With WandB logging
python trainer/pretrain.py --use_wandb --wandb_project "MyProject"

# With torch.compile optimization
python trainer/pretrain.py --use_compile 1
```

## Configuration Parameters

### Model Architecture
- `--hidden_size`: Hidden dimension (default: 512)
- `--num_hidden_layers`: Number of transformer layers (default: 8)
- `--max_seq_len`: Maximum sequence length (default: 340)
- `--use_moe`: Use Mixture of Experts (0/1, default: 0)

### Training
- `--epochs`: Training epochs (default: 1)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Initial learning rate (default: 5e-4)
- `--accumulation_steps`: Gradient accumulation steps (default: 8)
- `--grad_clip`: Gradient clipping threshold (default: 1.0)

### Data Loading
- `--data_path`: Path to JSONL training data
- `--num_workers`: DataLoader workers (default: 8)

### Mixed Precision
- `--dtype`: Mixed precision type (bfloat16/float16, default: bfloat16)

### Checkpointing
- `--save_dir`: Directory for model outputs (default: ../out)
- `--save_weight`: Save weight prefix (default: pretrain)
- `--from_weight`: Load pre-trained weights (default: none)
- `--from_resume`: Auto-detect and resume from checkpoint (0/1, default: 0)

### Monitoring
- `--log_interval`: Logging interval steps (default: 100)
- `--save_interval`: Checkpoint save interval (default: 1000)
- `--use_wandb`: Enable WandB logging (flag)
- `--wandb_project`: WandB project name (default: myLLM-Pretrain)

## Architecture Highlights

### Rotary Position Embeddings (RoPE)
- Context-aware position encoding that scales well to long sequences
- Optional YaRN scaling for context extension beyond training length

### Multi-Head Attention with KV Cache
- Efficient generation via cached key-value pairs
- Group Query Attention reduces memory for inference
- Flash Attention acceleration (PyTorch 2.0+)

### Efficient FFN
- Gating mechanism(SiLU activation)
- Aligned intermediate dimensions for GPU efficiency

### RMSNorm
- More efficient than LayerNorm
- Standard in modern LLMs (Llama, PaLM, etc.)

## Mixed Precision Training

- **bf16 (bfloat16)**: Better numerical stability, recommended for most cases
- **fp16 (float16)**: Faster but requires careful gradient scaling
- Automatic gradient scaling with torch.amp.GradScaler

## Distributed Training (DDP)

Automatically configures when using `torchrun`:
```bash
torchrun --nproc_per_node=4 trainer/pretrain.py
```

Environment variables handled automatically:
- RANK: Global process rank
- LOCAL_RANK: Local GPU rank
- MASTER_ADDR/MASTER_PORT: Coordination

## Checkpointing Strategy

**Two checkpoint files created:**

1. `{weight}_{hidden_size}.pth`: Model weights only (fp16 for efficiency)
2. `{weight}_{hidden_size}_resume.pth`: Complete training state for resumption
   - Model state dict
   - Optimizer state
   - Gradient scaler state
   - Current epoch/step
   - WandB run ID (for logging continuation)

**Atomic writes prevent corruption during save**

## Code Quality Features

### Comprehensive Documentation
- Module docstrings explaining purpose and usage
- Detailed function docstrings with Args/Returns
- Inline comments explaining key operations
- Type hints for better IDE support

### Best Practices
- Proper error handling and device compatibility
- Memory efficiency (fp16 weights, proper cleanup)
- Reproducibility via fixed random seeds
- Clean separation of concerns between modules

### Distributed Training Safe
- Main process guards for logging/checkpointing
- Proper DDP synchronization
- Handles world-size changes gracefully

## Performance Optimization

1. **Mixed Precision**: Reduces memory and speeds up computation
2. **Gradient Accumulation**: Effective larger batch sizes with limited VRAM
3. **Flash Attention**: ~3x faster attention computation
4. **torch.compile**: Optional JIT compilation for additional speedup
5. **Efficient Data Loading**: Multi-worker data loading with pin_memory
6. **Weight Sharing**: Embeddings and output layer share weights

## Example Training Session

```bash
# Start training on 4 GPUs
torchrun --nproc_per_node=4 trainer/pretrain.py \
    --epochs 2 \
    --batch_size 32 \
    --accumulation_steps 4 \
    --learning_rate 5e-4 \
    --use_wandb \
    --wandb_project "myLLM" \
    --data_path "../dataset/pretrain_hq.jsonl"

# Resume from checkpoint (auto-detects latest)
python trainer/pretrain.py --from_resume 1

# Evaluate model on same GPUs
python trainer/pretrain.py --epochs 0  # Just load and evaluate
```

## Dependencies

- PyTorch >= 2.0 (for Flash Attention)
- Transformers (tokenizer and modeling utilities)
- Datasets (for efficient data loading)
- einops (tensor operations)
- NumPy
- Optional: WandB (for experiment tracking)

## Future Extensions

- [ ] Mixture of Experts (MoE) full implementation
- [ ] CPU inference support
- [ ] Quantization (INT8, INT4)
- [ ] LoRA fine-tuning
- [ ] Multi-node training examples
- [ ] Generation utilities (beam search, etc.)

---

**Created with detailed documentation and best practices in mind for learning and production use.**
