import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Multi-scale convolutions with proper padding
        self.conv_scales = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.hidden_size, 
                out_channels=config.hidden_size // 4,
                kernel_size=2**i + 1,
                padding='same',  # Use 'same' padding to maintain sequence length
                stride=1
            ) for i in range(4)
        ])
        
        # Cross-scale attention for feature interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,  # Use 8 heads since it divides 896
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion and normalization
        self.fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Temperature prediction (now outputs hidden_size dimension)
        self.temp_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )
        
        # Move all modules to the correct dtype
        if hasattr(config, 'torch_dtype'):
            self.to(dtype=config.torch_dtype)

    def to(self, *args, **kwargs):
        """Override to method to ensure proper device/dtype propagation"""
        super().to(*args, **kwargs)
        # Ensure conv layers are properly moved
        if 'dtype' in kwargs:
            self.conv_scales = self.conv_scales.to(dtype=kwargs['dtype'])
        return self

    def forward(self, x):
        B, T, C = x.size()
        
        # Process at multiple scales
        multi_scale = []
        x_conv = x.transpose(1, 2)  # [B, C, T]
        
        # Ensure conv layers are on same device as input
        if next(self.conv_scales.parameters()).device != x.device:
            self.conv_scales = self.conv_scales.to(device=x.device)
        
        for conv in self.conv_scales:
            # Apply convolution and ensure output size matches input
            scale_feat = conv(x_conv.contiguous())
            
            # Ensure all features have the same sequence length
            if scale_feat.size(-1) != T:
                scale_feat = F.interpolate(scale_feat, size=T, mode='linear', align_corners=False)
            
            multi_scale.append(scale_feat)
        
        # Combine scales and reshape back
        combined = torch.cat(multi_scale, dim=1)  # [B, C*4, T]
        combined = combined.transpose(1, 2)  # [B, T, C*4]
        
        # Cross-scale attention
        attn_out, _ = self.cross_attention(x, combined, combined)
        
        # Fuse original and multi-scale features
        fused = self.fusion(torch.cat([x, attn_out], dim=-1))
        enhanced = self.norm(fused)
        
        # Predict temperatures with scale information (now same size as hidden_states)
        temps = F.softplus(self.temp_net(torch.cat([enhanced, attn_out], dim=-1))) + 0.5
        
        return enhanced, temps


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_length=2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create position encodings
        position = torch.arange(0, max_seq_length).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        
        # Register buffer
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Apply rotary embeddings
        seq_len = x.size(1)
        return x * self.pe[:seq_len].unsqueeze(0)

class MultiScaleTemperature(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # Multi-scale analysis
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU()
            ) for _ in range(3)  # 3 different scales
        ])
        
        # Temperature prediction per head
        self.temp_proj = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, config.num_attention_heads)
        )
    
    def forward(self, x):
        # Process at different scales
        scale_features = [scale(x) for scale in self.scales]
        combined = torch.cat(scale_features, dim=-1)
        
        # Predict temperatures
        temps = torch.sigmoid(self.temp_proj(combined))
        return temps

class TokenClusterProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # Projections for attention-based clustering
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        
        self.scale = self.d_model ** -0.5
        
    def forward(self, x, mask=None):
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Get attention weights and weighted values
        weights = F.softmax(scores, dim=-1)
        clustered = torch.matmul(weights, v)
        
        return clustered, weights

class OptimalPathSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.hidden_size
        
        # Path scoring network
        self.path_scorer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, 4)  # 4 different reasoning paths
        )
        
    def forward(self, x):
        # Score each reasoning path
        path_scores = self.path_scorer(x.mean(dim=1))  # Pool sequence
        path_probs = F.softmax(path_scores, dim=-1)
        
        # Select best path
        best_path = torch.argmax(path_probs, dim=-1)
        return best_path, path_probs

class GuidedReasoning(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Number of attention heads should divide hidden size evenly
        self.num_attention_heads = 8  # Fixed to 8 since it divides 896
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Multi-level reasoning components with progressive compression
        self.reasoning_levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            ) for _ in range(3)
        ])
        
        # Memory attention for each level
        self.memory_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=self.num_attention_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)
        ])
        
        # Level gates for dynamic weighting
        self.level_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.GELU(),
                nn.Linear(config.hidden_size // 4, 1),
                nn.Sigmoid()
            ) for _ in range(3)
        ])
        
        # Temperature prediction for each level (now outputs hidden_size dimension)
        self.temp_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, config.hidden_size)
            ) for _ in range(3)
        ])
        
        # Compression ratios for each level (1/2, 1/4, 1/8)
        self.compression_ratios = [2, 4, 8]
        
        self.final_fusion = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)

    def compress_sequence(self, x, ratio):
        """Compress sequence length by the given ratio"""
        B, T, C = x.size()
        target_len = max(1, T // ratio)
        
        # Use adaptive pooling for compression
        compressed = F.adaptive_avg_pool1d(
            x.transpose(1, 2),  # [B, C, T]
            target_len
        ).transpose(1, 2)  # [B, T', C]
        
        return compressed

    def expand_sequence(self, x, target_length):
        """Expand sequence back to target length"""
        B, T, C = x.size()
        if T == target_length:
            return x
            
        # Use linear interpolation for smooth expansion
        expanded = F.interpolate(
            x.transpose(1, 2),  # [B, C, T]
            size=target_length,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # [B, target_length, C]
        
        return expanded

    def forward(self, hidden_states):
        batch_size, seq_length = hidden_states.size()[:2]
        
        # Process each reasoning level
        level_outputs = []
        temperatures = []
        
        current_state = hidden_states
        for level, (ratio, reasoning_layer, memory_attn, gate, temp_net) in enumerate(zip(
            self.compression_ratios, self.reasoning_levels, self.memory_attention, 
            self.level_gates, self.temp_nets)):
            
            # Compress sequence for this level
            compressed = self.compress_sequence(current_state, ratio)
            
            # Apply reasoning transformation
            reasoned = reasoning_layer(compressed)
            
            # Memory attention at compressed length
            mem_out, _ = memory_attn(reasoned, reasoned, reasoned)
            
            # Expand back to original length
            expanded = self.expand_sequence(mem_out, seq_length)
            
            # Calculate gate value and apply
            gate_value = gate(expanded)
            gated_output = expanded * gate_value
            
            # Temperature prediction
            temps = F.softplus(temp_net(torch.cat([expanded, gated_output], dim=-1))) + 0.5
            
            level_outputs.append(gated_output)
            temperatures.append(temps)
            
            # Update current state with gated output
            current_state = gated_output
        
        # Combine gated outputs from all levels
        combined_output = torch.cat(level_outputs, dim=-1)
        final_output = self.norm(self.final_fusion(combined_output))
        
        # Average temperatures
        final_temps = torch.stack(temperatures).mean(dim=0)
        
        return final_output, final_temps

class EnhancedTokenImportance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Memory mechanism for token importance
        self.memory_size = 64
        self.memory = nn.Parameter(torch.randn(1, self.memory_size, config.hidden_size))
        
        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Guided reasoning with hierarchical structure
        self.guided_reasoning = GuidedReasoning(config)
        
        # Context processing with multi-scale features
        self.context_processor = ContextProcessor(config)
        self.rope = RotaryPositionalEmbedding(config.hidden_size, max_seq_length=config.max_position_embeddings)
        
        # Enhanced fusion with memory integration
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),  # Increased for reasoning
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)  # Outputs hidden_size dim
        )
        
        # Dynamic importance network
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1)  # Single importance score per token
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize with small weights
        with torch.no_grad():
            for layer in self.importance_net:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x, freq_indices=None, positions=None):
        batch_size = x.size(0)
        
        # Normalize input
        x = self.norm(x)
        
        # Query memory bank
        memory = self.memory.expand(batch_size, -1, -1)
        mem_out, _ = self.memory_attention(x, memory, memory)
        
        # Process with enhanced components
        context_out, context_temps = self.context_processor(x)
        reasoning_out, reasoning_temps = self.guided_reasoning(x)
        
        # Apply rotary embeddings if positions provided
        if positions is not None:
            x = self.rope(x)
        
        # Clear intermediate memory
        torch.cuda.empty_cache()
        
        # Combine all features
        combined = torch.cat([
            x,
            context_out * context_temps,  # Now matches hidden_size
            reasoning_out * reasoning_temps,  # Now matches hidden_size
            mem_out
        ], dim=-1)
        
        # Compute importance scores with memory context
        importance = self.importance_net(
            torch.cat([x, mem_out], dim=-1)
        ).squeeze(-1)  # [B, T]
        
        # Apply sigmoid to get scores between 0 and 1
        importance = torch.sigmoid(importance)
        
        # Final fusion with all components
        fused = self.fusion(combined)  # [B, T, H]
        
        # Clear intermediate memory
        torch.cuda.empty_cache()
        
        # Return with complete stats for monitoring
        stats = {
            'memory_attention': mem_out.mean().item(),
            'importance_scores': importance.mean().item(),
            'context_temps': context_temps.mean().item(),
            'reasoning_temps': reasoning_temps.mean().item(),
            'reasoning_stats': {
                'output_mean': reasoning_out.mean().item(),
                'temp_mean': reasoning_temps.mean().item()
            }
        }
        
        return importance, fused, stats

class SparseDenseAttention(nn.Module):
    """
    Ultra memory-efficient hybrid attention using:
    - Flash attention style computation
    - Gradient checkpointing
    - Aggressive memory management
    """
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        self.config = config
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size
        self.kv_heads = getattr(config, 'kv_heads', config.num_attention_heads)  
        self.head_size = config.hidden_size // config.num_attention_heads
        self.dropout = config.hidden_dropout_prob
        
        # Key, Query, Value projections with GQA support
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size * self.kv_heads // self.n_head, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size * self.kv_heads // self.n_head, bias=config.bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Register buffer for attention scaling
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(self.head_size)))

    def _compute_importance(self, q, k):
        # Compute raw importance scores
        importance = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # Scale importance scores
        importance = importance * self.config.importance_scale
        
        # Apply minimum threshold
        importance = torch.maximum(importance, torch.tensor(self.config.min_importance, device=importance.device))
        
        if self.training and self.config.importance_dropout > 0:
            # Apply dropout to importance scores
            mask = torch.bernoulli(torch.full_like(importance, 1 - self.config.importance_dropout))
            importance = importance * mask / (1 - self.config.importance_dropout)
        
        return importance

    def _chunk_attention(self, q, k, v, importance_chunk, chunk_size, temperature=None):
        B, H, L, D = q.shape
        
        # Process in smaller sub-chunks for memory efficiency
        out = torch.zeros_like(q)
        normalizer = torch.zeros(B, H, L, 1, device=q.device)
        
        for i in range(0, k.size(2), chunk_size):
            end_idx = min(i + chunk_size, k.size(2))
            
            # Get current key/value sub-chunk
            k_sub = k[:, :, i:end_idx]
            v_sub = v[:, :, i:end_idx]
            
            # Compute attention scores with numerical stability
            scores = torch.matmul(q[:, :, :chunk_size], k_sub.transpose(-2, -1))
            scores = scores * self.scale
            
            # Apply temperature scaling with stability checks
            if temperature is not None:
                temp = temperature.view(B, H, 1, 1)
                temp = torch.clamp(temp, min=0.1, max=100.0)  
                scores = scores / temp
            
            # Apply sparse attention with stability checks
            if importance_chunk is not None:
                imp = importance_chunk.view(B, H, -1, 1)
                imp = torch.sigmoid(imp)  
                imp = imp.expand(-1, -1, -1, end_idx - i)
                
                # Soft masking instead of hard masking
                mask_weight = torch.sigmoid((imp - 0.5) * 10)  
                scores = scores * mask_weight
                
                if hasattr(self.config, 'sparse_topk'):
                    topk = min(self.config.sparse_topk, scores.size(-1))
                    top_scores, _ = torch.topk(scores, k=topk, dim=-1)
                    min_top_score = top_scores[..., -1:]
                    scores = scores * (scores >= min_top_score).float()
            
            # Apply softmax with numerical stability
            max_score = torch.max(scores, dim=-1, keepdim=True)[0]
            scores = scores - max_score
            scores = torch.exp(scores)
            denom = torch.sum(scores, dim=-1, keepdim=True).clamp(min=1e-6)
            scores = scores / denom
            
            # Apply dropout
            scores = self.attn_dropout(scores)
            
            # Update output with scaled addition
            out = out + torch.matmul(scores, v_sub)
            normalizer = normalizer + scores.sum(dim=-1, keepdim=True)
        
        # Normalize with stability checks
        normalizer = torch.clamp(normalizer, min=1e-6)
        out = out / normalizer
        
        # Handle any remaining NaN values
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out

    def forward(self, x, importance_scores, temperatures=None):
        B, T, C = x.shape
        
        # Project to Q,K,V with stability
        qkv = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        qkv = torch.nan_to_num(qkv, nan=0.0, posinf=1.0, neginf=-1.0)
        q = qkv
        
        # Reshape to [B, H, T, D]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, C // self.kv_heads).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, C // self.kv_heads).transpose(1, 2)
        
        # Process attention in chunks
        chunk_size = min(T, 64)  
        num_chunks = (T + chunk_size - 1) // chunk_size
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        if temperatures is not None:
            temperatures = temperatures.view(B, self.n_head)
        
        # Process chunks
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, T)
            
            imp_chunk = importance_scores[:, start_idx:end_idx].view(B, -1, self.n_head).transpose(1, 2)
            
            chunk_output = self._chunk_attention(
                q[:, :, start_idx:end_idx],
                k,
                v,
                imp_chunk,
                end_idx - start_idx,
                temperatures
            )
            
            # Reshape and store chunk output
            chunk_output = chunk_output.transpose(1, 2).contiguous().view(B, end_idx - start_idx, C)
            output[:, start_idx:end_idx] = chunk_output
        
        # Final projection and dropout
        output = self.o_proj(output)
        output = self.resid_dropout(output)
        return output

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = SparseDenseAttention(config)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias),
            c_proj  = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias),
            act     = nn.GELU(),
            dropout = nn.Dropout(config.hidden_dropout_prob),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))
        
        # Initialize with smaller values
        for module in [self.mlp.c_fc, self.mlp.c_proj]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, importance_scores, temperatures=None):
        # Pre-norm architecture with residual scaling
        h = self.ln_1(x)
        
        # Self-attention with residual connection and scaling
        attn_out = self.attn(h, importance_scores, temperatures)
        if torch.isnan(attn_out).any():
            print("NaN detected in attention output")
            # Try to recover by using input
            h = x
        else:
            # Scale residual connection
            h = x + 0.1 * attn_out
        
        # MLP with pre-norm
        h2 = self.ln_2(h)
        mlp_out = self.mlpf(h2)
        
        if torch.isnan(mlp_out).any():
            print("NaN detected in MLP output")
            # Try to recover by using pre-MLP state
            out = h
        else:
            # Scale residual connection
            out = h + 0.1 * mlp_out
        
        # Ensure outputs are finite
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out

class DTATTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        
        # Position embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.hidden_size))
        
        # Enhanced token importance with guided reasoning
        self.importance = EnhancedTokenImportance(config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params}")
        
        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing = True

    def get_block_size(self):
        return self.config.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use Kaiming initialization for linear layers
            torch.nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity='leaky_relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use truncated normal for embeddings
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm with small values
            torch.nn.init.constant_(module.weight, 1.0)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, freq_indices=None, positions=None):
        device = idx.device
        b, t = idx.size()
        
        def check_nan(tensor, name):
            if torch.isnan(tensor).any():
                print(f"NaN detected in {name}")
                return True
            return False
        
        # Token embeddings
        tok_emb = self.wte(idx)
        if check_nan(tok_emb, "token embeddings"):
            return None, float('nan'), {}
            
        x = self.drop(tok_emb)
        
        # Get importance scores and temperatures
        importance_scores, temperatures, reasoning_stats = self.importance(x, freq_indices, positions)
        if check_nan(importance_scores, "importance_scores") or check_nan(temperatures, "temperatures"):
            return None, float('nan'), {}
        
        # Apply transformer blocks with gradient checkpointing if enabled
        if self.gradient_checkpointing and self.training:
            for i, block in enumerate(self.blocks):
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    importance_scores,
                    temperatures,
                    use_reentrant=False
                )
                if check_nan(x, f"block {i} output"):
                    return None, float('nan'), {}
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, importance_scores, temperatures)
                if check_nan(x, f"block {i} output"):
                    return None, float('nan'), {}
        
        # Final layer norm
        x = self.ln_f(x)
        if check_nan(x, "final layer norm"):
            return None, float('nan'), {}
        
        # Get logits
        logits = F.linear(x, self.wte.weight)
        if check_nan(logits, "logits"):
            return None, float('nan'), {}
            
        loss = None
        if targets is not None:
            # Apply label smoothing
            n_classes = logits.size(-1)
            smoothing = 0.1
            confidence = 1.0 - smoothing
            logits_flat = logits.view(-1, n_classes)
            targets_flat = targets.view(-1)
            
            # Create smoothed labels
            with torch.no_grad():
                true_dist = torch.zeros_like(logits_flat)
                true_dist.fill_(smoothing / (n_classes - 1))
                true_dist.scatter_(1, targets_flat.unsqueeze(1), confidence)
            
            # Compute loss with numerical stability
            log_probs = F.log_softmax(logits_flat, dim=-1)
            if check_nan(log_probs, "log_probs"):
                return None, float('nan'), {}
                
            loss = -(true_dist * log_probs).sum(-1).mean()
            if check_nan(loss, "loss before regularization"):
                return None, float('nan'), {}
            
            # Add L2 regularization for stability
            l2_lambda = 1e-4
            l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
            loss = loss + l2_lambda * l2_reg
            
            if check_nan(loss, "final loss"):
                return None, float('nan'), {}
        
        return logits, loss, {
            'importance_scores': importance_scores,
            'temperatures': temperatures,
            'reasoning_stats': reasoning_stats
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _, stats = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)



class DTATAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Initialize components
        self.context_processor = ContextProcessor(config)
        self.guided_reasoning = GuidedReasoning(config)
        self.token_importance = EnhancedTokenImportance(config)
        
        # Multi-head attention with proper scaling
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Initialize with small weights
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
        
        # Feature fusion
        self.feature_fusion = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=False)
        self.final_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.feature_fusion.weight, gain=0.02)
            torch.nn.init.xavier_uniform_(self.final_fusion.weight, gain=0.02)
        
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.scale = nn.Parameter(torch.ones(1) * 0.01)
        self.dropout = nn.Dropout(0.1)
        
        # Register scaling factor for attention
        self.register_buffer(
            "attention_scale",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).reciprocal()
        )
        
        # Move to correct dtype if specified
        if hasattr(config, 'torch_dtype'):
            self.to(dtype=config.torch_dtype)

    def forward(self, hidden_states):
        # Ensure input is in correct dtype
        dtype = hidden_states.dtype
        device = hidden_states.device
        
        batch_size, seq_len, _ = hidden_states.size()
        
        # Process context and reasoning first
        with torch.cuda.amp.autocast(enabled=True):
            context_out, context_temps = self.context_processor(hidden_states)
            reasoning_out, reasoning_temps = self.guided_reasoning(hidden_states)
            importance = self.token_importance(hidden_states)
            
            # Ensure all tensors have correct dtype
            context_out = context_out.to(dtype=dtype)
            context_temps = context_temps.to(dtype=dtype)
            reasoning_out = reasoning_out.to(dtype=dtype)
            reasoning_temps = reasoning_temps.to(dtype=dtype)
            importance = importance.to(dtype=dtype)
            
            # Ensure all tensors have correct shape
            importance = importance.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            
            # Apply weighting
            context_weighted = context_out * context_temps
            reasoning_weighted = reasoning_out * reasoning_temps
            token_weighted = hidden_states * importance
            
            # Combine features
            combined_features = torch.cat([
                context_weighted,
                reasoning_weighted,
                token_weighted,
                hidden_states
            ], dim=-1)
            
            # Feature fusion
            fused_features = self.feature_fusion(combined_features)
            
            # Multi-head attention processing
            q = self.q_proj(hidden_states).view(
                batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(fused_features).view(
                batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(fused_features).view(
                batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape and project output
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            # Final fusion with residual connection
            final_combined = torch.cat([hidden_states, attn_output], dim=-1)
            output = self.final_fusion(final_combined)
            output = self.layer_norm(output)
            output = hidden_states + self.scale * self.dropout(output)
            
            # Ensure output has same dtype as input
            output = output.to(dtype=dtype)
        
        return output

