"""
Language Model Dataset
=====================
This module provides dataset utilities for pretraining language models.

Key Components:
- PreTrainDataset: Efficient dataset class for loading and tokenizing text data
- Support for multi-file datasets using HuggingFace datasets library
- Automatic padding and special token handling
- Memory-efficient processing of large JSONL files
"""

import json
import torch
import os
import random
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Tuple, Dict, List


class PreTrainDataset(Dataset):
    """
    Dataset class for pre-training language models on raw text data.
    
    Loads text from JSONL files, tokenizes it, and prepares it for causal
    language modeling with proper handling of special tokens (BOS/EOS/PAD).
    
    Features:
    - Loads datasets from JSONL files via HuggingFace datasets library
    - Lazy loading for memory efficiency with large datasets
    - Automatic truncation to max_seq_len
    - Proper masking of padding tokens in labels
    
    Args:
        data_path: Path to JSONL file(s) containing text data
        tokenizer: Pre-trained tokenizer for encoding text
        max_seq_len: Maximum sequence length for samples
    
    Example:
        >>> dataset = PreTrainDataset('data.jsonl', tokenizer, max_seq_len=512)
        >>> input_ids, labels = dataset[0]
    """
    
    def __init__(self, data_path: str, tokenizer, max_seq_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load dataset from JSONL file using HuggingFace datasets
        # This supports multiple files matching the pattern in data_path
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Processing:
        1. Extract text from the data
        2. Tokenize with truncation (reserve 2 tokens for BOS/EOS)
        3. Add special tokens: [BOS] + tokens + [EOS]
        4. Pad to max_seq_len with PAD token
        5. Create labels with PAD tokens masked as -100 (ignored in loss)
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of:
                - input_ids: Token sequence of shape (max_seq_len,)
                - labels: Target sequence of shape (max_seq_len,) with PAD masked
        """
        # Get text content from the dataset
        text = self.samples[idx]['text']
        
        # Tokenize text with truncation
        # Reserve 2 tokens for BOS and EOS special tokens
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            add_special_tokens=False,  # We'll add BOS/EOS manually for consistency
            max_length=self.max_seq_len - 2, 
        )
        
        # Extract token IDs
        input_ids = encoding['input_ids']
        
        # Build final token sequence: [BOS] + tokens + [EOS] + paddi
        tokens = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        
        # Pad to max_seq_len
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))
        
        # Convert to tensor with dtype=int64 for compatibility with embedding layers
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Create labels tensor (same as input_ids initially)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels with -100 so they're ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Return input_ids and labels (attention_mask not used in current training loop)
        return input_ids, labels