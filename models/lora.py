"""
LoRA (Low-Rank Adaptation) Implementation for MMM Models

This module provides LoRA layers for parameter-efficient fine-tuning.
Based on: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    LoRA linear layer that adds low-rank adaptation to existing linear layers.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank matrices (default: 4)
        alpha: Scaling factor (default: 1)
        dropout: Dropout rate (default: 0.0)
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        """Compute LoRA adaptation: x @ (A @ B) * scaling"""
        return self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Wrapper that combines a frozen linear layer with LoRA adaptation.
    
    Args:
        linear: Original nn.Linear layer (will be frozen)
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
    """
    def __init__(self, linear, rank=4, alpha=1, dropout=0.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)


def add_lora_to_model(model, target_modules=None, rank=4, alpha=1, dropout=0.0):
    """
    Add LoRA to all Linear layers in a model (or specific target modules).
    
    Args:
        model: The model to add LoRA to
        target_modules: List of module name patterns to apply LoRA (e.g., ['q', 'v'])
                       If None, applies to all Linear layers
        rank: LoRA rank
        alpha: LoRA alpha  
        dropout: LoRA dropout
        
    Returns:
        Modified model with LoRA layers
    """
    def should_add_lora(name):
        if target_modules is None:
            return True
        return any(target in name for target in target_modules)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_add_lora(name):
            # Get parent and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                
            # Replace with LoRA version
            lora_linear = LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_linear)
            
    return model


def get_lora_params(model):
    """Get only LoRA parameters for optimization."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def merge_lora_weights(model):
    """
    Merge LoRA weights back into the original linear layers.
    This creates a single model without LoRA layers.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LinearWithLoRA):
            # Merge: W_new = W_original + A @ B * scaling
            with torch.no_grad():
                lora_weight = module.lora.lora_A @ module.lora.lora_B * module.lora.scaling
                merged_weight = module.linear.weight.data + lora_weight.T
                
            # Create new linear layer with merged weights
            merged_linear = nn.Linear(
                module.linear.in_features,
                module.linear.out_features,
                bias=module.linear.bias is not None
            )
            merged_linear.weight.data = merged_weight
            if module.linear.bias is not None:
                merged_linear.bias.data = module.linear.bias.data
                
            # Replace in model
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                
            setattr(parent, attr_name, merged_linear)
            
    return model


def save_lora_weights(model, save_path):
    """Save only LoRA parameters."""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.data
    torch.save(lora_state_dict, save_path)
    

def load_lora_weights(model, load_path):
    """Load LoRA parameters."""
    lora_state_dict = torch.load(load_path, map_location='cpu')
    model.load_state_dict(lora_state_dict, strict=False)
    return model


# Example usage:
if __name__ == "__main__":
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    )
    
    # Add LoRA
    model = add_lora_to_model(model, rank=8, alpha=16)
    
    # Get LoRA parameters for optimizer
    lora_params = get_lora_params(model)
    optimizer = torch.optim.Adam(lora_params, lr=1e-4)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable (LoRA) parameters: {sum(p.numel() for p in lora_params)}")
    
    # Forward pass
    x = torch.randn(4, 512)
    y = model(x)
    print(f"Output shape: {y.shape}")
