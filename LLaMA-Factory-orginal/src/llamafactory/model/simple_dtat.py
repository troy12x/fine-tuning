import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Create inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Create position embeddings
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        
        # Scaled time
        t = torch.arange(seq_len, device=device, dtype=dtype)
        t = t / self.scaling_factor
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Compute rotation matrices
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

class ConceptHierarchy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_levels = 3
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_attention_heads
        
        # Memory-efficient embeddings using sparse tensors
        self.level_concepts = nn.ParameterList([
            nn.Parameter(torch.zeros(config.vocab_size, config.hidden_size).normal_(0, 0.02))
            for _ in range(self.num_levels)
        ])
        
        # Concept enhancement layers
        self.concept_layers = nn.ModuleList([
            nn.ModuleDict({
                'extractor': nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.hidden_size * 4,
                    batch_first=True,
                    dropout=0.1,
                    norm_first=True
                ),
                'enhancer': nn.Sequential(
                    nn.Linear(config.hidden_size * 2, config.hidden_size),
                    nn.LayerNorm(config.hidden_size),
                    nn.GELU()
                )
            }) for _ in range(self.num_levels)
        ])
        
        # Cross-level enhancement
        self.cross_enhance = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(self.num_levels - 1)
        ])
        
        # Enhancement factors for each level
        self.level_factors = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1))
            for _ in range(self.num_levels)
        ])
        self.sigmoid = nn.Sigmoid()
        
        self.norm = nn.LayerNorm(config.hidden_size)
    
    def get_concept_embeddings(self, input_ids: torch.Tensor, level: int) -> torch.Tensor:
        return F.embedding(input_ids, self.level_concepts[level], sparse=True)
    
    def forward(self, x: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        # Store original input
        original_x = x
        
        # Process each level
        current_x = x
        all_concepts = []
        enhancement_factors = []
        
        for level in range(self.num_levels):
            # Get concepts
            concepts = self.get_concept_embeddings(input_ids, level)
            
            # Extract features
            features = self.concept_layers[level]['extractor'](
                current_x,
                src_key_padding_mask=~attention_mask if attention_mask is not None else None
            )
            
            # Generate enhancement
            enhancement = self.concept_layers[level]['enhancer'](
                torch.cat([features, concepts], dim=-1)
            )
            
            # Compute level-specific enhancement factor
            level_factor = self.sigmoid(self.level_factors[level])
            enhancement_factors.append(level_factor.item())
            
            # Enhanced state = current + weighted enhancement
            current_x = current_x + level_factor * enhancement
            current_x = self.norm(current_x)
            
            all_concepts.append(current_x)
            
            # Cross-level enhancement
            if level < self.num_levels - 1:
                cross_enhanced, _ = self.cross_enhance[level](
                    current_x, current_x, current_x,
                    key_padding_mask=~attention_mask if attention_mask is not None else None
                )
                current_x = current_x + 0.1 * cross_enhanced  # Small fixed factor for cross-enhancement
        
        # Final enhancement combining all levels
        final_enhancement = torch.stack(all_concepts, dim=1).mean(dim=1)
        final_enhanced = original_x + 0.1 * final_enhancement  # Conservative final enhancement
        
        return final_enhanced, {
            'level_factors': enhancement_factors,
            'num_levels': self.num_levels
        }

class RelationalGraphBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Edge type embeddings
        self.num_edge_types = 8
        self.edge_embeddings = nn.Parameter(
            torch.zeros(self.num_edge_types, config.hidden_size)
        )
        nn.init.normal_(self.edge_embeddings, std=0.02 / math.sqrt(self.num_edge_types))
        
        # Edge enhancement projection
        self.edge_enhance = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # Edge scoring with stability
        self.edge_scorer = nn.Sequential(
            nn.LayerNorm(config.hidden_size * 3),
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, self.num_edge_types)
        )
        
        # Graph propagation layers
        self.graph_enhance = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(
                    embed_dim=config.hidden_size,
                    num_heads=self.num_heads,
                    dropout=0.1,
                    batch_first=True
                ),
                nn.LayerNorm(config.hidden_size)
            ) for _ in range(2)
        ])
        
        # Enhancement factor
        self.enhance_factor = nn.Parameter(torch.tensor(0.1))
        self.sigmoid = nn.Sigmoid()
        
        # Numerical stability
        self.eps = 1e-6
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, nodes, mask=None):
        # Store original nodes
        original_nodes = nodes
        
        # 1. Build edge representations
        edges = self.build_edges(nodes, mask)
        
        # 2. Enhance nodes through graph
        enhanced_nodes = self.propagate(nodes, edges, mask)
        
        # 3. Compute enhancement factor
        enhance_weight = self.sigmoid(self.enhance_factor)
        
        # 4. Enhanced state = original + weighted enhancement
        final_nodes = original_nodes + enhance_weight * enhanced_nodes
        
        return edges, final_nodes
    
    def build_edges(self, nodes, mask=None):
        B, T, H = nodes.size()
        
        # Compute pairwise node features
        q = nodes.unsqueeze(2).expand(-1, -1, T, -1)  # [B, T, T, H]
        k = nodes.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, T, H]
        
        # Edge features
        edge_features = torch.cat([
            q,                      # Source node
            k,                      # Target node
            q * k / self.scale     # Interaction
        ], dim=-1)
        
        # Score edge types
        edge_logits = self.edge_scorer(edge_features)  # [B, T, T, num_edge_types]
        edge_probs = F.softmax(edge_logits, dim=-1)
        
        # Weight edge embeddings
        weighted_edges = torch.einsum(
            'btne,eh->btnh',
            edge_probs,
            self.edge_embeddings
        )
        
        # Enhance edges
        enhanced_edges = self.edge_enhance(weighted_edges)
        
        if mask is not None:
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, T, T]
            enhanced_edges = enhanced_edges.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        return enhanced_edges
    
    def propagate(self, nodes, edges, mask=None):
        # Initial state
        current = nodes
        
        # Propagate through layers
        for layer in self.graph_enhance:
            # Use edges to guide attention
            edge_weights = torch.einsum(
                'bthd,bshd->bths',
                current / self.scale,
                edges
            )
            
            # Apply attention with edge guidance
            enhanced, _ = layer[0](
                current, current, current,
                attn_mask=edge_weights,
                key_padding_mask=~mask if mask is not None else None
            )
            
            # Residual connection and norm
            current = current + enhanced
            current = layer[1](current)
        
        return current

class LogicalReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Reasoning transformer
        self.reasoning_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size * 4,
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )
        
        # Enhancement projection
        self.enhance = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
    
    def forward(self, x, temps, mask=None):
        # Apply reasoning attention
        reasoned = self.reasoning_layer(x, src_key_padding_mask=~mask if mask is not None else None)
        
        # Enhance with temperature-weighted attention
        enhanced = self.enhance(torch.cat([
            x * temps,  # Temperature-weighted input
            reasoned   # Reasoned state
        ], dim=-1))
        
        return enhanced

class SemanticProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.concept_hierarchy = ConceptHierarchy(config)
        self.graph_builder = RelationalGraphBuilder(config)
        
        # Enhancement fusion (not replacement)
        self.semantic_enhancement = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
        
        # Learnable enhancement factor
        self.enhance_factor = nn.Parameter(torch.tensor(0.15))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, input_ids, attention_mask=None):
        # Store original input
        original_x = x
        
        # 1. Get concepts with proper dimensions
        concepts, concept_info = self.concept_hierarchy(x, input_ids, attention_mask)
        concepts = self.concept_proj(concepts)  # Ensure correct dimension
        
        # 2. Build graph with proper dimensions
        edges, graph_state = self.graph_builder(concepts, attention_mask)
        graph_state = self.graph_proj(graph_state)  # Ensure correct dimension
        
        # Verify dimensions
        assert original_x.size(-1) == concepts.size(-1) == graph_state.size(-1), \
            f"Dimension mismatch: {original_x.size(-1)}, {concepts.size(-1)}, {graph_state.size(-1)}"
        
        # Combine all semantic information
        semantic_features = torch.cat([
            original_x,    # [batch, seq, hidden]
            concepts,      # [batch, seq, hidden]
            graph_state   # [batch, seq, hidden]
        ], dim=-1)  # Result: [batch, seq, hidden*3]
        
        # 4. Generate enhancement with correct output dimension
        semantic_enhancement = self.semantic_enhancement(semantic_features)  # Back to [batch, seq, hidden]
        
        # 5. Small, fixed enhancement
        enhanced_state = original_x + 0.1 * semantic_enhancement  # Fixed small enhancement
        
        return enhanced_state, {
            'concepts': concept_info,
            'edges': edges,
            'graph_state': graph_state
        }

class EnhancedReasoning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.semantic = SemanticProcessor(config)
        self.reasoner = LogicalReasoner(config)
        
        # Temperature computation
        self.temperature = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Enhancement integration (not replacement)
        self.enhancement = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
    
    def forward(self, x, input_ids, attention_mask=None):
        # Store original input
        original_x = x
        
        # 1. Process semantics
        semantic_state, semantic_info = self.semantic(
            x, input_ids, attention_mask
        )
        
        # 2. Compute temperatures for attention
        temps = self.temperature(torch.cat([
            original_x, semantic_state
        ], dim=-1))
        
        # 3. Apply reasoner
        reasoned_state = self.reasoner(semantic_state, temps, attention_mask)
        
        # 4. Create enhancement
        enhancement = self.enhancement(
            torch.cat([original_x, reasoned_state], dim=-1)
        )
        
        # Return enhanced state and metadata
        return enhancement, {
            'semantic': semantic_info,
            'temperatures': temps
        }

class SimpleReasoningLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reasoning = EnhancedReasoning(config)
    
    def forward(self, x, input_ids, attention_mask=None):
        return self.reasoning(x, input_ids, attention_mask)

class TokenTemperatureMechanism(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # Temperature projection matrices with increased dimensionality
        self.W_t = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        self.temperature_mha = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Enhanced temperature dynamics
        self.update_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Token type emphasis
        self.type_emphasis = nn.Parameter(torch.ones(4))  # [math, operator, variable, text]
        
        # Stochastic effects with controlled scale
        self.noise_scale = nn.Parameter(torch.tensor(0.005))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Handle unbatched input
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        # Enhanced projection for temperature computation
        temp_query = self.W_t(x)
        
        # Multi-head attention for temperature (TTM)
        # For MHA, mask should be True for valid positions
        attention_mask = mask if mask is None else ~mask
        temp_out, _ = self.temperature_mha(temp_query, x, x, 
                                         key_padding_mask=attention_mask)
        
        # Combine features with residual connection
        combined = torch.cat([x * self.type_emphasis[0], temp_out * self.type_emphasis[1]], dim=-1)
        
        # Temperature computation with enhanced dynamics
        temperatures = self.update_net(combined)
        
        # Add controlled noise during training
        if self.training:
            noise = torch.randn_like(temperatures) * self.noise_scale
            temperatures = temperatures + noise
        
        # Normalize temperatures
        temperatures = temperatures / temperatures.max()
        
        # Remove batch dimension if input was unbatched
        if x.size(0) == 1:
            temperatures = temperatures.squeeze(0)
        
        return temperatures

class MultiScaleTemperatureAnalysis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_scales = 3
        self.hidden_size = config.hidden_size
        
        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(self.num_scales)
        ])
        
        # Coupling factors
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1))
            for _ in range(self.num_scales - 1)
        ])
        
    def forward(self, x, prev_temps=None):
        scale_temps = []
        
        for s in range(self.num_scales):
            if s == 0:
                # Base scale
                temp = F.sigmoid(self.scale_projections[s](x))
            else:
                # Higher scales with neighborhood influence
                neighbor_influence = self.gamma[s-1] * prev_temps
                temp = F.sigmoid(self.scale_projections[s](x) + neighbor_influence)
            
            scale_temps.append(temp)
            prev_temps = temp
            
        return torch.stack(scale_temps, dim=1)

class ReasoningPathOptimizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Path weight network
        self.path_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        
        # Initialize weights based on layer dimensions
        for m in self.path_net.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.size(1)
                fan_out = m.weight.size(0)
                std = math.sqrt(2.0 / (fan_in + fan_out))
                gain = nn.init.calculate_gain('leaky_relu')  # More stable than relu
                nn.init.normal_(m.weight, mean=0.0, std=std * gain)
                if m.bias is not None:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
        
        # Dynamic backtracking threshold based on scale count
        self.register_buffer('scale_count', torch.tensor(3, dtype=torch.float))
    
    def forward(self, scale_states, temperatures):
        """Compute path weights and backtracking signal
        
        Args:
            scale_states: Multi-scale hidden states [batch, num_scales, seq, hidden]
            temperatures: Temperature values [batch, num_scales, seq, 1]
        """
        batch_size, num_scales, seq_len, hidden_size = scale_states.shape
        
        # Update scale count if needed
        if num_scales != self.scale_count:
            self.scale_count.fill_(float(num_scales))
        
        # Reshape temperatures if needed
        if temperatures.dim() == 5:
            temperatures = temperatures.squeeze(1)
        if temperatures.shape[-1] == hidden_size:
            temperatures = temperatures.mean(dim=-1, keepdim=True)
        
        # Normalize temperatures using robust statistics
        temp_mean = temperatures.mean(dim=(0,2), keepdim=True)
        temp_std = temperatures.std(dim=(0,2), keepdim=True) + torch.finfo(temperatures.dtype).eps
        temperatures = (temperatures - temp_mean) / temp_std
        
        # Average scale states and temperatures across sequence
        avg_states = scale_states.mean(dim=2)  # [batch, num_scales, hidden]
        avg_temps = temperatures.mean(dim=2)   # [batch, num_scales, 1]
        
        # Layer normalize states per scale with eps
        avg_states = F.layer_norm(avg_states.transpose(1,2), (num_scales,), eps=1e-5).transpose(1,2)
        
        # Normalize temperatures more robustly
        temp_stats = avg_temps.view(-1)  # Flatten for stable stats
        temp_mean = temp_stats.mean()
        temp_std = temp_stats.std() + 1e-5
        avg_temps = (avg_temps - temp_mean) / temp_std
        
        # Expand temperatures to match hidden size with gradient scaling
        avg_temps = avg_temps.expand(-1, -1, hidden_size) * 0.1  # Scale factor to prevent domination
        
        # Compute path weights for each scale
        combined = torch.cat([avg_states, avg_temps], dim=-1)
        combined = F.dropout(combined, p=0.1, training=self.training)  # Add dropout for regularization
        flat_combined = combined.view(-1, hidden_size * 2)
        
        # Get path logits with stable normalization
        path_logits = self.path_net(flat_combined)
        path_logits = path_logits.view(batch_size, num_scales, 1)
        
        # Normalize logits for stable softmax
        logit_stats = path_logits.view(-1)  # Flatten for stable stats
        logit_mean = logit_stats.mean()
        logit_std = logit_stats.std() + 1e-5
        path_logits = (path_logits - logit_mean) / logit_std
        
        # Compute path weights with temperature scaling
        path_weights = F.softmax(path_logits, dim=1)
        path_weights = F.dropout(path_weights, p=0.1, training=self.training)
        
        # Compute backtracking threshold dynamically
        weight_diffs = (path_weights[:, :-1, :] - path_weights[:, 1:, :])
        diff_mean = weight_diffs.mean()
        diff_std = weight_diffs.std() + torch.finfo(weight_diffs.dtype).eps
        should_backtrack = diff_mean > diff_std
        
        return path_weights, should_backtrack

class MultiScaleReasoning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_scales = 3
        self.hidden_size = config.hidden_size
        
        # Scale-specific transformations
        self.scale_transforms = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(self.num_scales)
        ])
        
        # Inter-scale coupling
        self.scale_couplings = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1))
            for _ in range(self.num_scales - 1)
        ])
        
        # Temperature projection matrices
        self.W_s = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(self.num_scales)
        ])
        
    def forward(self, x, base_temperatures, attention_mask=None):
        scale_outputs = []
        current_temp = base_temperatures
        
        for s in range(self.num_scales):
            # Transform at current scale
            scale_hidden = self.scale_transforms[s](
                x,
                src_key_padding_mask=~attention_mask if attention_mask is not None else None
            )
            
            if s > 0:
                # Apply inter-scale coupling
                neighbor_influence = self.scale_couplings[s-1] * scale_outputs[-1]
                scale_hidden = scale_hidden + neighbor_influence
            
            # Update temperatures for this scale
            scale_temp = F.sigmoid(
                self.W_s[s](scale_hidden) + 
                (current_temp if s > 0 else 0)
            )
            
            scale_outputs.append(scale_hidden)
            current_temp = scale_temp
        
        return torch.stack(scale_outputs, dim=1), current_temp

class TemperatureGuidedAttention(nn.Module):
    """Temperature-guided attention combining TTM + GSoT with Qwen-2 architecture."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # Qwen-2 attention projections (bias=False as per Qwen-2)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # TTM components from paper
        self.ttm_mha = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout=config.attention_dropout,
            bias=False,  # Qwen-2 style
            batch_first=True
        )
        self.W_t = nn.Linear(self.hidden_size, self.num_attention_heads, bias=False)  # W_t from paper
        self.ttm_norm = nn.LayerNorm(self.num_attention_heads)  # For numerical stability
        
        # Temperature projection to match hidden size
        self.temp_hidden_proj = nn.Sequential(
            nn.Linear(self.num_attention_heads, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
        # GSoT components from paper
        self.num_scales = 3  # As per paper
        rope_base = getattr(config, "rope_theta", None)  # Try to get model-specific base
        if rope_base is None:
            # Fallback to checking other common config names
            rope_base = getattr(config, "rotary_base", None)
            if rope_base is None:
                rope_base = getattr(config, "rope_base", 10000.0)  # Default if not specified
                
        self.R = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # R(x) transformation
        self.scale_weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.hidden_size))  # w_s
            for _ in range(self.num_scales)
        ])
        self.scale_betas = nn.ParameterList([
            nn.Parameter(torch.ones(1))  # β_s
            for _ in range(self.num_scales)
        ])
        self.scale_gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1))  # γ_s
            for _ in range(self.num_scales - 1)
        ])
        
        # Temperature dynamics (from paper)
        self.temp_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),  # Input: concatenated hidden states and temperature
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),  # Output: single temperature value
            nn.Sigmoid()
        )
        
        # Qwen-2 RoPE with dynamic NTK
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=rope_base,
            scaling_factor=config.rope_scaling.get("factor", 1.0) if hasattr(config, "rope_scaling") else 1.0
        )
        
        # Qwen-2 parallel layers
        self.parallel_attention = config.parallel_attention if hasattr(config, "parallel_attention") else False
        if self.parallel_attention:
            self.parallel_mlp = QwenMLP(config)
        
        # Other components
        self.attn_scale = self.head_dim ** -0.5
        self.noise_scale = nn.Parameter(torch.tensor(0.01))  # η_l scale
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
        self._init_weights()
    
    def compute_ttm(self, x, mask=None, step=0):
        """Implements T(x) = σ(W_t · MHA(x) + b_t) from paper with safety bounds."""
        # Handle mask dimensions for MHA
        if mask is not None and mask.dim() == 1 and x.dim() == 3:
            mask = mask.unsqueeze(0)  # Add batch dimension to match input
            
        # Multi-head attention for temperature (TTM)
        ttm_out, _ = self.ttm_mha(x, x, x, key_padding_mask=mask)
        
        # Project and normalize (W_t from paper)
        temp_logits = self.W_t(ttm_out)
        temp = self.ttm_norm(temp_logits)
        
        # Add noise during training (η_l term from paper) with safety bound
        if self.training:
            noise_scale = min(self.noise_scale * (0.9 ** step), 0.01)  # Cap maximum noise
            noise = torch.randn_like(temp) * noise_scale
            temp = temp + noise
        
        # Ensure temperature stays close to 1 initially
        temp = F.sigmoid(temp)
        if step < 100:  # Early training stability
            temp = 1.0 + 0.1 * (temp - 0.5)  # Center around 1.0 with small deviation
        
        return temp
        
    def compute_gsot(self, hidden_states, T_0):
        """Compute Guided Sequence of Thought temperatures"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        scale_temps = []
        
        # Project initial temperature
        T_base = self.temp_hidden_proj(T_0)
        T_base = torch.sigmoid(T_base)  # Bound between 0 and 1
        scale_temps.append(T_base[:, :, 0:1])
        
        # Compute temperatures for each scale
        for i in range(1, 3):
            hidden_flat = hidden_states.view(-1, hidden_size)
            T_base_flat = T_base.view(-1, hidden_size)
            
            combined = torch.cat([hidden_flat, T_base_flat], dim=-1)
            T_i = self.temp_net(combined)
            T_i = torch.sigmoid(T_i)  # Simple bounding
            T_i = T_i.view(batch_size, seq_len, -1)
            
            scale_temps.append(T_i)
            T_base = T_i.expand(-1, -1, hidden_size)
        
        return torch.stack(scale_temps, dim=1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Store step for scaling factors
        self.training_steps = step
        
        # Compute initial temperature (TTM) with safety
        T_0 = self.compute_ttm(hidden_states, attention_mask, step)
        
        # Compute GSoT temperatures with bounds
        scale_temps = self.compute_gsot(hidden_states, T_0)  # [batch, num_scales, seq, 1]
        
        # Average temperatures across scales and reshape for attention
        temp_weights = scale_temps.mean(dim=1)  # [batch, seq, 1]
        temp_weights = temp_weights.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)  # [batch, heads, seq, 1]
        temp_weights = temp_weights.expand(-1, -1, -1, seq_length)  # [batch, heads, seq, seq]
        
        # Ensure temp_weights stay close to 1 with limited deviation
        temp_weights = 1.0 + (temp_weights - temp_weights.mean()) / (temp_weights.std() + torch.finfo(temp_weights.dtype).eps)
        
        # Original Qwen attention path
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)
        
        # Split heads and transpose: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        query_states = query_states.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores with temperature weighting
        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.attn_scale
        
        # Apply temperature weights
        attention_scores = attention_scores * temp_weights
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert 1D/2D mask to attention mask [batch, seq, seq]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
            attention_mask = attention_mask.expand(-1, self.num_attention_heads, seq_length, -1)  # [batch, heads, seq, seq]
            
            # Convert boolean mask to float mask
            attention_mask = attention_mask.to(dtype=attention_scores.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Get output
        context = torch.matmul(attention_probs, value_states)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_length, -1)
        context = self.o_proj(context)
        
        # Residual safety for parallel MLP
        if self.parallel_attention:
            mlp_output = self.parallel_mlp(hidden_states)
            # Gradual mixing of attention and MLP outputs
            mix_ratio = min(1.0, step / 1000) if self.training else 1.0
            context = (1 - mix_ratio) * context + mix_ratio * mlp_output
        
        if output_attentions:
            outputs = (context, attention_probs)
        else:
            outputs = (context,)
            
        if use_cache:
            outputs += ((key_states, value_states),)
        
        # Return both context and scale temperatures
        return context, scale_temps
    
    def _init_weights(self):
        """Initialize weights using Qwen-2 scheme."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

class EnhancedGuidedSequenceOfThought(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_scales = 3  # Number of reasoning scales
        
        # Multi-scale reasoning components
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU()
            ) for _ in range(self.num_scales)
        ])
        
        # Temperature-guided attention
        self.temp_attention = TemperatureGuidedAttention(config)
        
        # Path optimization
        self.path_optimizer = ReasoningPathOptimizer(config)
        
        # Output transformation
        self.output_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Multi-scale reasoning
        scale_states = []
        for scale_transform in self.scale_transforms:
            scale_state = scale_transform(hidden_states)
            scale_states.append(scale_state)
        
        # Stack scale states [batch, num_scales, seq_len, hidden]
        scale_states = torch.stack(scale_states, dim=1)
        
        # Get temperatures from attention mechanism
        attn_output, temperatures = self.temp_attention(hidden_states, attention_mask)
        
        # Ensure temperatures have correct shape and are finite
        temperatures = temperatures.view(batch_size, -1, seq_len, 1)  # [batch, num_scales, seq, 1]
        
        # Optimize reasoning paths
        path_weights, should_backtrack = self.path_optimizer(
            scale_states, temperatures
        )
        
        # Combine states using path weights
        path_weights = path_weights.unsqueeze(-1)  # [batch, num_scales, 1, 1]
        weighted_states = torch.sum(scale_states * path_weights, dim=1)  # [batch, seq_len, hidden]
        
        # Transform output
        output = self.output_transform(weighted_states)
        
        # Return output and auxiliary info
        return output, {
            'scale_states': scale_states.detach(),  # Detach to avoid memory leaks
            'path_weights': path_weights.squeeze(-1).detach(),  # [batch, num_scales, 1]
            'temperatures': temperatures.detach(),  # [batch, num_scales, seq, 1]
            'should_backtrack': should_backtrack.detach()  # Scalar
        }

class SimpleDTAT(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Temperature mechanism
        self.ttm = TokenTemperatureMechanism(config)
        
        # Enhanced guided reasoning
        self.gsot = EnhancedGuidedSequenceOfThought(config)
        
        # DTAT processor
        self.dtat_processor = SimpleReasoningLayer(config)
        
        # Blend factors
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.temp_factor = nn.Parameter(torch.tensor(0.15))
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info_rank0(f"Enhanced DTAT initialized with {total_params:,} parameters")
        
        # Set compute dtype if specified
        if hasattr(config, 'compute_dtype'):
            self.to(config.compute_dtype)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def train(self, mode: bool = True):
        """Set training mode for DTAT and base model."""
        super().train(mode)
        self.base_model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode for DTAT and base model."""
        super().eval()
        self.base_model.eval()
        return self

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get base model output
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = base_outputs.last_hidden_state
        
        # Apply temperature mechanism
        temperatures = self.ttm(hidden_states, attention_mask)
        
        # Apply guided reasoning with temperature
        gsot_states = self.gsot(hidden_states, temperatures, attention_mask)
        
        # Apply DTAT processing
        dtat_states = self.dtat_processor(hidden_states, input_ids, attention_mask)
        
        # Controlled integration with temperature guidance
        temp_factor = self.sigmoid(self.temp_factor)
        alpha = self.sigmoid(self.alpha)
        
        # Combine enhancements with temperature-guided mixing
        enhanced_states = (
            (1 - temp_factor) * dtat_states +
            temp_factor * gsot_states
        )
        
        # Final blending
        final_states = (1 - alpha) * hidden_states + alpha * enhanced_states
        
        # Return with original structure
        base_outputs.last_hidden_state = final_states
        return base_outputs
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def _reorder_cache(self, *args, **kwargs):
        return self.base_model._reorder_cache(*args, **kwargs)
    
    @property
    def device(self):
        return self.base_model.device
