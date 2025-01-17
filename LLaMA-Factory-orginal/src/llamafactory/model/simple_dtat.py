import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
    
    def forward(self, x: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

class SimpleDTAT(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Add DTAT enhancement layer
        self.dtat_processor = SimpleReasoningLayer(config)
        
        # Blend factor - learnable parameter
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.sigmoid = nn.Sigmoid()
        
        # Initialize only our new weights
        self.dtat_processor.apply(self._init_weights)
        
        # Count parameters
        dtat_params = sum(p.numel() for p in self.dtat_processor.parameters())
        print(f"DTAT enhancement initialized with {dtat_params:,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get base model output
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = base_outputs.last_hidden_state
        
        # Apply DTAT processing
        enhanced_states = self.dtat_processor(hidden_states, input_ids, attention_mask)
        
        # Blend with original (controlled enhancement)
        alpha = self.sigmoid(self.alpha)
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
