import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleReasoningLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Token importance scoring with careful initialization
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),  # Add normalization for stability
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize importance network with small weights
        for module in self.importance_net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Reasoning components
        self.reasoning_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Memory-efficient attention
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dynamic temperature prediction
        self.temp_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),  # Add normalization for stability
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, self.num_heads)
        )
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Register buffer for attention scaling
        self.register_buffer(
            "attention_scale",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).reciprocal()
        )

    def forward(self, x, attention_mask=None):
        identity = x  # Save input for residual connection
        B, T, C = x.shape
        
        # Compute token importance scores with safe computation
        importance = self.importance_net(x)  # [B, T, 1]
        importance = importance.clamp(min=1e-6)  # Prevent zero importance
        
        # Apply reasoning transformation with importance weighting
        reasoned = self.reasoning_net(x * importance)  # Scale input by importance
        
        # Project queries, keys, values
        q = self.q_proj(reasoned)  # Use reasoned representations for queries
        k = self.k_proj(x * importance)  # Scale keys by importance
        v = self.v_proj(x)  # Keep original values
        
        # Reshape for attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Dynamic temperature with safe computation
        temp = F.softplus(self.temp_net(reasoned.mean(dim=1)))  # [B, num_heads]
        temp = temp.unsqueeze(-1).unsqueeze(-1) + 0.5  # [B, num_heads, 1, 1]
        
        # Compute attention scores with masking support
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        
        if attention_mask is not None:
            # Expand mask for heads: [B, 1, T, T]
            attention_mask = attention_mask.unsqueeze(1)
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        
        attn = attn * temp  # Apply temperature scaling
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        out = self.o_proj(out)
        
        # Residual connection and normalization
        return self.norm(identity + out)

class SimpleDTAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        
        # Reasoning layers
        self.reasoning_layers = nn.ModuleList([
            SimpleReasoningLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # Output normalization
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None):
        x = self.wte(input_ids)
        x = self.drop(x)
        
        # Apply reasoning layers with attention mask
        for layer in self.reasoning_layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    attention_mask
                )
            else:
                x = layer(x, attention_mask)
        
        x = self.ln_f(x)
        return x

    def clip_gradients(self):
        """Helper method to clip gradients for stable training"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
