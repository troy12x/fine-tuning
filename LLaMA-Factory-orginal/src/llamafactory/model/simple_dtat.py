import math
from typing import Optional, Tuple, Dict, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptHierarchy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_levels = getattr(config, 'num_concept_levels', 3)
        self.num_heads = config.num_attention_heads
        
        # Concept embeddings per level (using smaller vocabulary)
        self.level_concepts = nn.ParameterList([
            nn.Parameter(torch.randn(2 ** (level + 1), config.hidden_size))
            for level in range(self.num_levels)
        ])
        
        # Initialize with small values
        for param in self.level_concepts:
            nn.init.normal_(param, std=0.02)
        
        # Concept enhancement layers (from paper)
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
        
        # Cross-level enhancement (from paper)
        self.cross_enhance = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(self.num_levels - 1)
        ])
        
        # Enhancement factors (β_k in paper)
        self.level_factors = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1))
            for _ in range(self.num_levels)
        ])
        self.sigmoid = nn.Sigmoid()
        
        # Final combination
        self.concept_combine = nn.Sequential(
            nn.Linear(config.hidden_size * self.num_levels, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
    
    def get_concept_embeddings(self, hidden_states, level):
        """Get concept embeddings for a specific level using hidden states"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project hidden states to get concept logits
        concept_logits = torch.matmul(hidden_states, self.level_concepts[level].transpose(0, 1))
        
        # Get concept indices through argmax
        concept_indices = torch.argmax(concept_logits, dim=-1)  # [batch, seq]
        
        # Get embeddings for those concepts
        return F.embedding(concept_indices, self.level_concepts[level], sparse=True)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: Input hidden states [batch, seq, hidden]
            attention_mask: Optional attention mask [batch, seq]
        """
        # Store original input
        original_x = hidden_states
        
        # Process each level
        current_x = hidden_states
        all_concepts = []
        enhancement_factors = []
        
        for level in range(self.num_levels):
            # Get concepts using hidden states
            concepts = self.get_concept_embeddings(current_x, level)
            
            # Extract features
            features = self.concept_layers[level]['extractor'](
                current_x,
                src_key_padding_mask=~attention_mask if attention_mask is not None else None
            )
            
            # Generate enhancement
            enhancement = self.concept_layers[level]['enhancer'](
                torch.cat([features, concepts], dim=-1)
            )
            
            # Compute level-specific enhancement factor (β_k)
            level_factor = self.sigmoid(self.level_factors[level])
            enhancement_factors.append(level_factor.item())
            
            # Enhanced state = current + weighted enhancement
            current_x = current_x + level_factor * enhancement
            
            all_concepts.append(current_x)
            
            # Cross-level enhancement
            if level < self.num_levels - 1:
                cross_enhanced, _ = self.cross_enhance[level](
                    current_x, current_x, current_x,
                    key_padding_mask=~attention_mask if attention_mask is not None else None
                )
                current_x = current_x + 0.1 * cross_enhanced
        
        # Combine concepts from all levels
        combined = torch.cat(all_concepts, dim=-1)  # [batch, seq, hidden*num_levels]
        concepts = self.concept_combine(combined)  # [batch, seq, hidden]
        
        # Small residual connection
        enhanced = original_x + 0.1 * concepts
        
        return enhanced, {
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
            # Reshape for multi-head attention
            B, T, H = current.shape
            current_heads = current.view(B, T, self.num_heads, -1)  # [B, T, heads, head_dim]
            edges_heads = edges.view(B, T, T, self.num_heads, -1)  # [B, T, T, heads, head_dim]
            
            # Use edges to guide attention (fixed einsum)
            edge_weights = torch.einsum(
                'bthd,bsthd->bhts',  # Note: changed einsum pattern to match dimensions
                current_heads,        # [B, T, heads, head_dim]
                edges_heads          # [B, T, T, heads, head_dim]
            ) / self.scale
            
            # Convert 4D attention weights to 3D for MultiheadAttention
            # Original: [batch, heads, tgt_len, src_len] -> New: [batch * heads, tgt_len, src_len]
            edge_weights = edge_weights.view(B * self.num_heads, T, T)
            
            # Apply attention with edge guidance
            enhanced, _ = layer[0](
                current, current, current,
                attn_mask=edge_weights,
                key_padding_mask=~mask if mask is not None else None,
                average_attn_weights=False  # Don't average across heads
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
        self.concept_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Enhancement fusion (not replacement)
        self.semantic_enhancement = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),  # Combine original, concept, and graph states
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Numerical stability
        self.eps = 1e-6
        
    def forward(self, x, input_ids=None, attention_mask=None):
        """
        Args:
            x: Input hidden states [batch, seq, hidden]
            input_ids: Optional input token ids
            attention_mask: Optional attention mask
        """
        # Store original input
        original_x = x
        
        # 1. Extract hierarchical concepts
        concept_states, concept_info = self.concept_hierarchy(x, attention_mask)
        concept_states = self.concept_proj(concept_states)  # Project concept states
        
        # 2. Build graph with proper dimensions
        edges, graph_state = self.graph_builder(concept_states, attention_mask)
        
        # Verify dimensions
        assert original_x.size(-1) == concept_states.size(-1) == graph_state.size(-1), \
            f"Hidden size mismatch: {original_x.size(-1)}, {concept_states.size(-1)}, {graph_state.size(-1)}"
            
        # 3. Combine features
        semantic_features = torch.cat([
            original_x,           # Original sequence features
            concept_states,       # Hierarchical concept features  
            graph_state          # Graph-enhanced features
        ], dim=-1)  # [batch, seq, hidden*3]
        
        # 4. Generate semantic enhancement
        semantic_enhancement = self.semantic_enhancement(semantic_features)  # [batch, seq, hidden]
        
        # Small residual connection
        enhanced = original_x + 0.1 * semantic_enhancement
        
        return enhanced, {
            'concept_info': concept_info,
            'graph_state': graph_state,
            'edges': edges
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
        self.semantic = SemanticProcessor(config)
        self.reasoning = EnhancedReasoning(config)
        
    def forward(self, x, input_ids=None, attention_mask=None):
        """
        Args:
            x: Input hidden states [batch, seq, hidden]
            input_ids: Optional input token ids
            attention_mask: Optional attention mask
        """
        # Process through semantic and reasoning layers
        semantic_state, semantic_info = self.semantic(x, input_ids, attention_mask)
        reasoning_state, reasoning_info = self.reasoning(semantic_state, input_ids, attention_mask)
        
        # Combine info dictionaries
        info = {
            'semantic': semantic_info,
            'reasoning': reasoning_info
        }
        
        return reasoning_state, info

class TokenTemperatureMechanism(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_scales = getattr(config, 'num_scales', 3)
        self.noise_scale = 0.1
        
        # Temperature projection matrices with increased dimensionality
        self.W_t = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
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
        
        # Temperature stabilization
        self.temp_ema = None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Handle unbatched input
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        # Convert mask to float for attention
        if mask is not None:
            attention_mask = mask.to(dtype=torch.bool)
        
        # Project input to temperature space
        temp_proj = self.W_t(x)  # [batch, seq, hidden]
        
        # Apply temperature MHA with proper key padding mask
        if mask is not None:
            attention_mask = ~mask  # Invert mask since True means position to attend
        else:
            attention_mask = None
            
        temp_attn, _ = self.temperature_mha(temp_proj, temp_proj, temp_proj, 
                                          key_padding_mask=attention_mask)
        
        # Combine features with residual connection
        combined = torch.cat([x, temp_attn], dim=-1)
        
        # Temperature computation with enhanced dynamics
        temperatures = self.update_net(combined)  # [batch, seq, 1]
        
        # Add controlled noise during training
        if self.training:
            noise = torch.randn_like(temperatures) * self.noise_scale
            temperatures = temperatures + noise
            
        # Clamp and normalize temperatures
        temperatures = torch.clamp(temperatures, min=1e-6, max=1.0)
        
        # Normalize per sequence
        max_temps = temperatures.max(dim=1, keepdim=True)[0]
        temperatures = temperatures / (max_temps + 1e-6)
        
        # Handle unbatched case
        if x.dim() == 2:
            temperatures = temperatures.unsqueeze(0)
        
        # For analysis, return just [batch, seq, 1]
        if not hasattr(self, 'training') or not self.training:
            return temperatures
            
        # For training, expand to [batch, num_scales, seq, 1]
        temps_expanded = temperatures.unsqueeze(1).expand(-1, self.num_scales, -1, -1)
        return temps_expanded

class MultiScaleTemperatureAnalysis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_scales = getattr(config, 'num_scales', 3)
        self.hidden_size = config.hidden_size
        
        # Scale-specific projections with LayerNorm for stability
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, 1)
            ) for _ in range(self.num_scales)
        ])
        
        # Coupling factors
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1))
            for _ in range(self.num_scales - 1)
        ])
        
    def forward(self, x, prev_temps=None):
        batch_size, seq_len, hidden_size = x.shape
        scale_temps = []
        
        # Initialize prev_temps if None
        if prev_temps is None:
            prev_temps = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        for s in range(self.num_scales):
            if s == 0:
                # Base scale
                temp = torch.sigmoid(self.scale_projections[s](x))  # [batch, seq, 1]
            else:
                # Higher scales with neighborhood influence
                neighbor_influence = self.gamma[s-1] * prev_temps
                temp = torch.sigmoid(self.scale_projections[s](x) + neighbor_influence)
            
            scale_temps.append(temp)
            prev_temps = temp
        
        # Stack along scale dimension and permute to [batch, num_scales, seq, 1]
        temps = torch.stack(scale_temps, dim=1)  # [batch, num_scales, seq, 1]
        
        # Ensure proper shape and no NaN values
        temps = torch.clamp(temps, min=1e-6, max=1.0)
        temps = temps / (temps.max(dim=2, keepdim=True)[0] + 1e-6)
        
        return temps

class ReasoningPathOptimizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_scales = getattr(config, 'num_scales', 3)  # Default number of scales
        self.backtrack_threshold = getattr(config, 'backtrack_threshold', 0.5)
        self.path_dropout = getattr(config, 'path_dropout', 0.1)
        
        # Path weight network
        self.path_net = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.path_dropout),
            nn.Linear(self.hidden_size, 1)
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
    
    def forward(self, scale_states, temperatures):
        """Forward pass of path optimizer
        Args:
            scale_states: [batch_size, num_scales, seq_len, hidden_size]
            temperatures: [batch_size, num_scales, seq_len, 1] or [batch_size, 1, seq_len, 1]
        Returns:
            path_weights: [batch_size, num_scales, 1]
            should_backtrack: bool
        """
        batch_size, num_scales, seq_len, hidden_size = scale_states.shape
        
        # Update scale count if needed
        if num_scales != self.num_scales:
            self.num_scales = num_scales
            
        # Ensure temperatures have right shape and scale appropriately
        if temperatures.size(1) == 1:  # [batch, 1, seq, 1]
            temperatures = temperatures.expand(-1, num_scales, -1, -1)
            
        # Get statistics across sequence length
        avg_states = scale_states.mean(dim=2)  # [batch, num_scales, hidden]
        avg_temps = temperatures.mean(dim=2)  # [batch, num_scales, 1]
        
        # Combine state and temperature info
        combined = torch.cat([avg_states, avg_temps], dim=-1)  # [batch, num_scales, hidden+1]
        
        # Apply dropout during training
        combined = F.dropout(combined, p=self.path_dropout, training=self.training)
        
        # Reshape for path network
        combined = combined.view(-1, hidden_size + 1)  # [batch*num_scales, hidden+1]
        
        # Get path weights with proper normalization
        path_logits = self.path_net(combined)  # [batch*num_scales, 1]
        path_logits = path_logits.view(batch_size, num_scales, 1)  # [batch, num_scales, 1]
        
        # Apply temperature-scaled softmax
        path_weights = F.softmax(path_logits / math.sqrt(hidden_size), dim=1)
        path_weights = F.dropout(path_weights, p=self.path_dropout, training=self.training)
        
        # Compute backtracking signal
        temp_stats = temperatures.mean(dim=(0,1,2))  # Average over batch, scales, seq
        should_backtrack = temp_stats < self.backtrack_threshold
        
        return path_weights, should_backtrack

class MultiScaleReasoning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_scales = getattr(config, 'num_scales', 3)  # Number of reasoning scales
        
        # Temperature-guided attention for each scale
        self.temp_attention = TemperatureGuidedAttention(config)
        
        # Path optimizer
        self.path_optimizer = ReasoningPathOptimizer(config)
        
    def forward(self, hidden_states, temperatures, attention_mask=None):
        """Multi-scale reasoning forward pass
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            temperatures: [batch_size, seq_len, 1]
            attention_mask: Optional attention mask
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create multi-scale states
        scale_states = []
        for i in range(self.num_scales):
            # Apply temperature-guided attention for this scale
            scale_output = self.temp_attention(hidden_states, attention_mask)
            scale_states.append(scale_output)
            hidden_states = scale_output  # Use output as input for next scale
            
        # Stack scale states [batch, num_scales, seq, hidden]
        scale_states = torch.stack(scale_states, dim=1)
        
        # Create scaled temperatures for each reasoning scale
        temps_list = []
        for i in range(self.num_scales):
            # Scale temperature based on reasoning depth
            scale_factor = 1.0 / (1.0 + i * 0.5)  # Smoother decay: 1.0, 0.67, 0.5
            scale_temp = temperatures * scale_factor
            temps_list.append(scale_temp)
        scale_temps = torch.stack(temps_list, dim=1)  # [batch, num_scales, seq, 1]
        
        # Get path weights and backtracking signal
        path_weights, should_backtrack = self.path_optimizer(scale_states, scale_temps)
        
        # Apply path weights to scale states
        path_weights = path_weights.unsqueeze(2)  # [batch, num_scales, 1, 1]
        weighted_states = scale_states * path_weights
        gsot_states = weighted_states.sum(dim=1)  # [batch, seq, hidden]
        
        return gsot_states, {
            'path_weights': path_weights.squeeze(-1),  # [batch, num_scales, 1]
            'should_backtrack': should_backtrack,
            'scale_temps': scale_temps  # [batch, num_scales, seq, 1]
        }

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
        
        # Temperature projection layers
        self.temperature_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()  # Normalize to [0,1]
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Temperature projection to match hidden size
        self.temp_hidden_proj = nn.Sequential(
            nn.Linear(1, self.hidden_size // 4),  # Project from scalar temperature
            nn.LayerNorm(self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, self.hidden_size),  # Project to full hidden size
            nn.LayerNorm(self.hidden_size)
        )
        
        # GSoT components from paper
        self.num_scales = getattr(config, 'num_scales', 3)  # As per paper
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
        self.rotary_emb = None
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "rotary":
            self.rotary_emb = config.get_rotary_embedding(self.head_dim) if hasattr(config, "get_rotary_embedding") else None
        
        # Qwen-2 parallel layers
        self.parallel_attention = config.parallel_attention if hasattr(config, "parallel_attention") else False
        if self.parallel_attention:
            self.parallel_mlp = config.get_mlp() if hasattr(config, "get_mlp") else self.create_default_mlp(config)
        
        # Other components
        self.attn_scale = self.head_dim ** -0.5
        self.noise_scale = nn.Parameter(torch.tensor(0.01))  # η_l scale
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
        self._init_weights()
    
    def compute_ttm(self, query, key, value, step=None):
        """Compute temperature-guided token mixing"""
        batch_size, seq_len, hidden_size = query.shape
        
        # Get attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hidden_size)  # [batch, seq, seq]
        
        # Process each position's attention pattern
        scores_flat = scores.view(-1, seq_len)  # [batch*seq, seq]
        
        # Project to temperature values
        temp = self.temperature_proj(value)  # [batch, seq, 1]
        
        return temp
    
    def compute_gsot(self, hidden_states, T_0):
        """Compute Guided Sequence of Thought temperatures"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project initial temperature (process each position independently)
        T_0_flat = T_0.view(-1, 1)  # [batch*seq, 1]
        T_base = self.temp_hidden_proj(T_0_flat)  # [batch*seq, hidden]
        T_base = T_base.view(batch_size, seq_len, hidden_size)  # [batch, seq, hidden]
        T_base = torch.sigmoid(T_base)  # Bound between 0 and 1
        scale_temps = [T_base[:, :, 0:1]]  # Take first dimension for temperature
        
        # Compute temperatures for each scale
        for i in range(1, self.num_scales):
            hidden_flat = hidden_states.view(-1, hidden_size)
            T_base_flat = T_base.view(-1, hidden_size)
            
            combined = torch.cat([hidden_flat, T_base_flat], dim=-1)
            T_i = self.temp_net(combined)
            T_i = torch.sigmoid(T_i)  # Simple bounding
            T_i = T_i.view(batch_size, seq_len, 1)  # Make sure temperature is 1D
            
            scale_temps.append(T_i)
            T_base = T_i.expand(-1, -1, hidden_size)
        
        # Stack temperatures [batch, num_scales, seq, 1]
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
        T_0 = self.compute_ttm(hidden_states, hidden_states, hidden_states, step)
        
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
            mlp_output = self.parallel_mlp(hidden_states) if self.parallel_mlp else hidden_states
            # Gradual mixing of attention and MLP outputs
            mix_ratio = min(1.0, step / getattr(self.config, "warmup_steps", 1000)) if self.training else 1.0
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

    def create_default_mlp(self, config):
        """Creates a default MLP if not provided by config"""
        return nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.1)
        )

class EnhancedGuidedSequenceOfThought(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_scales = getattr(config, 'num_scales', 3)  # Number of reasoning scales
        self.hidden_size = config.hidden_size
        self.scale_dropout = getattr(config, 'scale_dropout', 0.1)
        
        # Multi-scale reasoning components
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(self.scale_dropout)
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
    
    def forward(self, hidden_states, temperatures, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Process each scale
        scale_states = []
        for i in range(self.num_scales):
            # Transform hidden states for this scale
            scale_state = self.scale_transforms[i](hidden_states)
            
            # Apply temperature-guided attention and get only the output state
            attended_state, _ = self.temp_attention(
                scale_state,
                attention_mask=attention_mask
            )
            scale_states.append(attended_state)
        
        # Stack scale states [batch, num_scales, seq_len, hidden]
        scale_states = torch.stack(scale_states, dim=1)
        
        # Ensure temperatures have correct shape
        if temperatures.dim() == 3:  # [batch, seq, 1]
            temperatures = temperatures.unsqueeze(1)  # [batch, 1, seq, 1]
        elif temperatures.dim() == 4:  # [batch, num_scales, seq, 1] or [num_scales, seq, 1, 1]
            if temperatures.size(0) != batch_size:
                # Handle case where batch dimension is not first
                temperatures = temperatures.permute(0, 1, 2, 3)
                if temperatures.size(0) != batch_size:
                    temperatures = temperatures.transpose(0, 1)  # Move batch to front
            
        # Validate shape
        assert temperatures.shape == (batch_size, self.num_scales, seq_len, 1), \
            f"Temperature shape mismatch: expected {(batch_size, self.num_scales, seq_len, 1)}, got {temperatures.shape}"
            
        # Get path weights and backtracking signal
        path_weights, should_backtrack = self.path_optimizer(
            scale_states, temperatures
        )
        
        # Weight and combine scale states
        path_weights = path_weights.unsqueeze(-1)  # [batch, num_scales, 1, 1]
        weighted_states = scale_states * path_weights
        combined_state = weighted_states.sum(dim=1)  # [batch, seq_len, hidden]
        
        # Final transformation
        output_state = self.output_transform(combined_state)
        
        return output_state, {
            'path_weights': path_weights.squeeze(-1).detach(),  # [batch, num_scales, 1]
            'should_backtrack': should_backtrack,
            'temperatures': temperatures.detach()  # [batch, num_scales, seq, 1]
        }

class SimpleDTAT(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.num_scales = getattr(config, 'num_scales', 3)
        
        # Get device and dtype from base model
        self._device = next(base_model.parameters()).device
        self._dtype = next(base_model.parameters()).dtype
        
        # Initialize components with matching dtype
        self.ttm = TokenTemperatureMechanism(config).to(device=self._device, dtype=self._dtype)
        self.gsot = EnhancedGuidedSequenceOfThought(config).to(device=self._device, dtype=self._dtype)
        self.dtat_processor = SimpleReasoningLayer(config).to(device=self._device, dtype=self._dtype)
        
        # Initialize blending parameters
        self.temp_factor = nn.Parameter(torch.tensor(0.5, device=self._device, dtype=self._dtype))
        self.alpha = nn.Parameter(torch.tensor(0.5, device=self._device, dtype=self._dtype))
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Enhanced DTAT initialized with {total_params:,} parameters")
        
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

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Move inputs to correct device and dtype
        if input_ids is not None:
            input_ids = input_ids.to(device=self._device)  # Keep ids as long
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self._device, dtype=torch.bool)  # Convert to bool
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype in [torch.long, torch.bool, torch.uint8]:
                    # Don't change dtype for special types
                    kwargs[k] = v.to(device=self._device)
                else:
                    kwargs[k] = v.to(device=self._device, dtype=self._dtype)
                
        # Get base model output
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = base_outputs.hidden_states[-1] if hasattr(base_outputs, 'hidden_states') else base_outputs.last_hidden_state
        
        # Temperature-guided enhancement
        temperatures = self.ttm(hidden_states, attention_mask)
        
        # Ensure temperatures have the right shape for multi-scale processing
        if temperatures.dim() == 3:  # [batch, seq, 1]
            temperatures = temperatures.unsqueeze(1)  # [batch, 1, seq, 1]
            temperatures = temperatures.expand(-1, self.num_scales, -1, -1)
        
        # Apply GSoT and DTAT processing
        gsot_states, gsot_info = self.gsot(hidden_states, temperatures, attention_mask)
        dtat_states, dtat_info = self.dtat_processor(hidden_states, input_ids, attention_mask)
        
        # Dynamic blending
        temp_factor = torch.sigmoid(self.temp_factor).to(dtat_states.dtype)
        alpha = torch.sigmoid(self.alpha).to(hidden_states.dtype)
        
        # Combine enhancements
        enhanced_states = (
            (1.0 - temp_factor.view(1, 1, 1)) * dtat_states +
            temp_factor.view(1, 1, 1) * gsot_states
        )
        
        # Final integration
        final_states = (1.0 - alpha.view(1, 1, 1)) * hidden_states + alpha.view(1, 1, 1) * enhanced_states
        
        # Update output with enhanced states
        if hasattr(base_outputs, 'hidden_states'):
            base_outputs.hidden_states = base_outputs.hidden_states[:-1] + (final_states,)
        else:
            base_outputs.last_hidden_state = final_states
            
        # Store auxiliary info for debugging
        base_outputs.gsot_info = gsot_info
        base_outputs.dtat_info = dtat_info
            
        return base_outputs
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def _reorder_cache(self, *args, **kwargs):
        return self.base_model._reorder_cache(*args, **kwargs)
    
    @property
    def device(self):
        return self.base_model.device
