import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class EfficientContextProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Simplified multi-scale with just 2 key scales
        self.conv_scales = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size // 2,
                kernel_size=2**i + 1,
                padding='same',
                groups=8  # Group convolution for efficiency
            ) for i in range(2)  # Reduced to 2 scales
        ])
        
        # Streamlined fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
    
    def forward(self, x):
        x_reshaped = x.transpose(1, 2)
        multi_scale = []
        for conv in self.conv_scales:
            scale_out = conv(x_reshaped)
            multi_scale.append(scale_out)
        
        combined = torch.cat(multi_scale, dim=1)
        return self.fusion(combined.transpose(1, 2))

class MemoryEfficientProcessor:
    """Memory-efficient processing utilities"""
    
    @staticmethod
    def chunk_forward(tensor, chunk_size, dim=0):
        """Process large tensors in chunks"""
        chunks = tensor.chunk(max(tensor.size(dim) // chunk_size, 1), dim=dim)
        return torch.cat([chunk for chunk in chunks], dim=dim)
    
    @staticmethod
    def efficient_attention(q, k, v, mask=None):
        """Memory-efficient attention implementation"""
        B, H, T, D = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.view(B, 1, 1, T), float('-inf'))
        
        # Compute attention in chunks to save memory
        attn_chunks = []
        chunk_size = 1024  # Adjust based on available memory
        
        for i in range(0, T, chunk_size):
            end_idx = min(i + chunk_size, T)
            chunk_scores = scores[:, :, i:end_idx, :]
            chunk_probs = torch.softmax(chunk_scores, dim=-1)
            chunk_output = torch.matmul(chunk_probs, v)
            attn_chunks.append(chunk_output)
        
        return torch.cat(attn_chunks, dim=2)

class ConceptHierarchy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_levels = 3
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_attention_heads
        
        # Validate config
        assert hasattr(config, 'vocab_size'), "Config must specify vocab_size"
        if hasattr(config, 'tokenizer_vocab_size'):
            assert config.vocab_size == config.tokenizer_vocab_size
        
        # Memory-efficient embeddings using sparse tensors
        self.level_concepts = nn.ParameterList([
            nn.Parameter(torch.zeros(config.vocab_size, config.hidden_size).normal_(0, 0.02))
            for _ in range(self.num_levels)
        ])
        
        # Concept layers with memory optimization
        self.concept_layers = nn.ModuleList([
            nn.ModuleDict({
                'extractor': nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.hidden_size * 4,
                    batch_first=True,
                    dropout=0.1,
                    norm_first=True  # More stable training
                ),
                'aggregator': nn.Sequential(
                    nn.Linear(config.hidden_size * 2, config.hidden_size),
                    nn.LayerNorm(config.hidden_size),
                    nn.GELU()
                ),
                'projector': nn.Linear(config.hidden_size, config.hidden_size)
            }) for _ in range(self.num_levels)
        ])
        
        # Cross-level attention with memory-efficient implementation
        self.cross_level_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(self.num_levels - 1)
        ])
        
        # Numerical stability
        self.eps = 1e-6
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize with proper scaling
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for name, p in self.named_parameters():
            if 'weight' in name:
                # Scaled initialization
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_levels))
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def get_concept_embeddings(self, input_ids: torch.Tensor, level: int) -> torch.Tensor:
        return F.embedding(input_ids, self.level_concepts[level], sparse=True)
    
    def forward(self, x: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, H = x.size()
        all_concepts = []
        level_x = x
        
        # Process each level
        for level in range(self.num_levels):
            # Get concepts
            concepts = self.get_concept_embeddings(input_ids, level)
            
            # Extract features
            features = self.concept_layers[level]['extractor'](
                level_x,
                src_key_padding_mask=~attention_mask if attention_mask is not None else None
            )
            
            # Project and combine
            concepts = self.concept_layers[level]['projector'](concepts)
            combined = self.concept_layers[level]['aggregator'](
                torch.cat([features, concepts], dim=-1)
            )
            
            # Add residual and normalize
            level_x = self.norm(combined + level_x)
            all_concepts.append(level_x)
            
            # Cross-level attention except for last level
            if level < self.num_levels - 1:
                level_x, _ = self.cross_level_attention[level](
                    level_x, level_x, level_x,
                    key_padding_mask=~attention_mask if attention_mask is not None else None
                )
        
        return torch.stack(all_concepts, dim=1)  # [B, num_levels, T, H]

class RelationalGraphBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Edge type embeddings with proper initialization
        self.num_edge_types = 8
        self.edge_embeddings = nn.Parameter(
            torch.zeros(self.num_edge_types, config.hidden_size)
        )
        nn.init.normal_(self.edge_embeddings, std=0.02 / math.sqrt(self.num_edge_types))
        
        # Edge type projection with layer norm for stability
        self.edge_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # Numerically stable edge scorer
        self.edge_scorer = nn.Sequential(
            nn.LayerNorm(config.hidden_size * 3),  # Normalize inputs
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),  # Intermediate normalization
            nn.Linear(config.hidden_size, self.num_edge_types)
        )
        
        # Graph propagation with stability improvements
        self.graph_prop = nn.ModuleList([
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
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Numerical stability
        self.eps = 1e-6
        self.scale = math.sqrt(self.head_dim)  # Proper scaling factor
    
    def build_edges(self, nodes, mask=None):
        """Numerically stable edge building"""
        B, T, H = nodes.size()
        
        # Project edge embeddings
        edge_emb = self.edge_proj(self.edge_embeddings)  # [E, H]
        
        # Compute node pair representations efficiently
        nodes_i = nodes.unsqueeze(2)  # [B, T, 1, H]
        nodes_j = nodes.unsqueeze(1)  # [B, 1, T, H]
        
        # Scale inputs for numerical stability
        nodes_i = nodes_i / self.scale
        nodes_j = nodes_j / self.scale
        edge_emb = edge_emb / self.scale
        
        # Efficient edge computation
        edge_inputs = torch.cat([
            nodes_i.expand(-1, -1, T, -1),  # [B, T, T, H]
            nodes_j.expand(-1, T, -1, -1),  # [B, T, T, H]
            edge_emb.unsqueeze(0).unsqueeze(0).expand(B, T, T, -1)  # [B, T, T, H]
        ], dim=-1)  # [B, T, T, 3H]
        
        # Compute edge scores with numerical stability
        edge_scores = self.edge_scorer(edge_inputs)  # [B, T, T, E]
        
        if mask is not None:
            # Create attention mask
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, T, T]
            edge_scores = edge_scores.masked_fill(
                ~mask_2d.unsqueeze(-1),
                float('-inf')
            )
        
        # Stable softmax with temperature scaling
        edge_scores = edge_scores / self.scale
        edge_probs = F.softmax(edge_scores, dim=-1)  # [B, T, T, E]
        edge_probs = self.dropout(edge_probs)
        
        return edge_probs
    
    def propagate(self, nodes, edges, mask=None):
        """Memory-efficient graph propagation"""
        B, T, H = nodes.size()
        x = nodes
        
        for prop_layer in self.graph_prop:
            # Compute attention weights
            edge_weights = edges.sum(dim=-1)  # [B, T, T]
            
            # Apply attention scaling
            edge_weights = edge_weights / (edge_weights.sum(dim=-1, keepdim=True) + self.eps)
            
            # Apply mask if provided
            if mask is not None:
                mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, T, T]
                edge_weights = edge_weights.masked_fill(~mask_2d, 0.0)
            
            # Reshape for attention
            q = x.view(B, T, H)  # [B, T, H]
            k = x.view(B, T, H)  # [B, T, H]
            v = x.view(B, T, H)  # [B, T, H]
            
            # Apply attention
            x_new, _ = prop_layer[0](
                q, k, v,
                attn_mask=edge_weights,
                key_padding_mask=mask
            )  # [B, T, H]
            
            # Apply layer norm
            x_new = prop_layer[1](x_new)  # [B, T, H]
            
            # Add residual and dropout
            x = x + self.dropout(x_new)
        
        return x

class LogicalReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Step generator with optimized architecture
        self.step_generator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size * 4,
                batch_first=True,
                norm_first=True  # Better gradient flow
            ),
            num_layers=2
        )
        
        # Optimized operator application
        self.num_operators = 8
        self.operators = nn.Parameter(
            torch.zeros(self.num_operators, config.hidden_size)
        )
        nn.init.normal_(self.operators, std=0.02 / math.sqrt(self.num_operators))
        
        # Operator projection with layer norm
        self.op_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # Optimized step validator
        self.step_validator = nn.Sequential(
            nn.LayerNorm(config.hidden_size * 3),
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Path scorer with stability
        self.path_scorer = nn.Sequential(
            nn.LayerNorm(config.hidden_size * 2),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1)
        )
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Enable TorchScript for performance
        self.use_script = True
        if self.use_script:
            self.step_validator = torch.jit.script(self.step_validator)
            self.path_scorer = torch.jit.script(self.path_scorer)
    
    def generate_step(self, state, temps, mask=None):
        """Optimized step generation"""
        B, T, H = state.size()
        
        # Apply temperature scaling
        weighted = state * temps.unsqueeze(-1)  # [B, T, H]
        
        # Generate steps efficiently
        steps = self.step_generator(weighted, src_key_padding_mask=mask)  # [B, T, H]
        
        # Project operators once
        ops = self.op_proj(self.operators)  # [O, H]
        
        # Efficient operator application
        steps = steps.unsqueeze(1)  # [B, 1, T, H]
        ops = ops.unsqueeze(0).unsqueeze(2)  # [1, O, 1, H]
        
        # Use broadcasting for efficiency
        op_steps = steps * ops  # [B, O, T, H]
        
        return self.norm(op_steps)
    
    def validate_step(self, step, prev_state):
        """Memory-efficient step validation"""
        B, O, T, H = step.size()
        
        # Reshape efficiently
        step_flat = step.view(-1, H)  # [B*O*T, H]
        prev_flat = prev_state.unsqueeze(1).expand(-1, O, -1, -1).reshape(-1, H)
        
        # Compute validity scores
        valid_input = torch.cat([
            step_flat,
            prev_flat,
            step_flat * prev_flat
        ], dim=-1)  # [B*O*T, 3H]
        
        validity = self.step_validator(valid_input)  # [B*O*T, 1]
        return validity.view(B, O, T, 1)
    
    def score_path(self, path_states):
        """Optimized path scoring"""
        B, S, T, H = path_states.size()
        
        # Pre-allocate scores tensor
        scores = torch.empty(B, S-1, T, 1, device=path_states.device)
        
        # Score adjacent steps efficiently
        for i in range(S - 1):
            # Get adjacent states
            curr_state = path_states[:, i].reshape(-1, H)  # [B*T, H]
            next_state = path_states[:, i+1].reshape(-1, H)  # [B*T, H]
            
            # Compute score
            score = self.path_scorer(
                torch.cat([curr_state, next_state], dim=-1)
            )  # [B*T, 1]
            
            scores[:, i] = score.view(B, T, 1)
        
        return scores

class SemanticProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.concept_hierarchy = ConceptHierarchy(config)
        self.graph_builder = RelationalGraphBuilder(config)
        
        # Semantic integration
        self.semantic_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
    
    def forward(self, x, input_ids, attention_mask=None):
        # 1. Build concept hierarchy
        concepts = self.concept_hierarchy(x, input_ids, attention_mask)
        
        # 2. Create relational graph
        edges = self.graph_builder.build_edges(concepts[-1], attention_mask)
        
        # 3. Propagate through graph
        graph_state = self.graph_builder.propagate(
            concepts[-1], edges, attention_mask
        )
        
        # 4. Integrate all semantic info
        semantic_state = self.semantic_fusion(torch.cat([
            x,
            concepts[0],  # Low-level concepts
            concepts[-1],  # High-level concepts
            graph_state
        ], dim=-1))
        
        return semantic_state, {
            'concepts': concepts,
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
        
        # 2. Compute temperatures (importance weights)
        temps = self.temperature(torch.cat([
            original_x, semantic_state
        ], dim=-1))
        
        # 3. Initialize reasoning from semantic state
        state = semantic_state
        all_states = []
        all_validities = []
        
        # 4. Iterative reasoning
        max_steps = 5
        min_validity = 0.8
        
        for step in range(max_steps):
            # Generate possible steps
            steps = self.reasoner.generate_step(
                state, temps, attention_mask
            )
            
            # Validate steps
            validities = torch.stack([
                self.reasoner.validate_step(step, state)
                for step in steps.unbind(1)
            ], dim=1)
            
            # Select best valid step
            best_idx = torch.argmax(validities, dim=1)
            best_step = torch.gather(
                steps, 1,
                best_idx.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, 1, steps.size(2), steps.size(3)
                )
            ).squeeze(1)
            
            # Accumulate enhancement (not replace)
            state = state + best_step
            
            # Store progress
            all_states.append(state)
            all_validities.append(validities)
            
            # Check if reasoning is sufficient
            if validities.mean() > min_validity:
                break
        
        # 5. Combine reasoning steps
        reasoning_states = torch.stack(all_states, dim=1)
        reasoning_validities = torch.stack(all_validities, dim=1)
        
        # Weight by validities
        weights = F.softmax(reasoning_validities, dim=1).unsqueeze(-1)
        reasoned_enhancement = (reasoning_states * weights).sum(1)
        
        # Create enhancement (not replacement)
        enhancement = self.enhancement(
            torch.cat([original_x, reasoned_enhancement], dim=-1)
        )
        
        # Return enhanced state and metadata
        return enhancement, {
            'semantic': semantic_info,
            'temperatures': temps,
            'reasoning_states': reasoning_states,
            'reasoning_validities': reasoning_validities,
            'num_steps': step + 1
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
        self.alpha = nn.Parameter(torch.tensor(0.2))  # Start with more weight on original
        self.sigmoid = nn.Sigmoid()  # Keep alpha between 0 and 1
        
        # Enhancement projection
        self.enhancement_proj = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
        
        # Initialize only our new weights
        self.dtat_processor.apply(self._init_weights)
        self.enhancement_proj.apply(self._init_weights)
        
        # Count only the new parameters we're adding
        dtat_params = sum(p.numel() for p in self.dtat_processor.parameters())
        proj_params = sum(p.numel() for p in self.enhancement_proj.parameters())
        print(f"DTAT enhancement initialized with {dtat_params + proj_params:,} additional parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get base model output - keeping original computation path intact
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
        original_hidden_states = base_outputs.last_hidden_state
        
        # Get DTAT enhancements
        dtat_enhanced, reasoning_info = self.dtat_processor(
            original_hidden_states,
            input_ids,
            attention_mask
        )
        
        # Compute dynamic blend factor
        alpha = self.sigmoid(self.alpha)
        
        # Combine original and enhanced states
        combined = self.enhancement_proj(
            torch.cat([original_hidden_states, dtat_enhanced], dim=-1)
        )
        
        # Enhanced hidden state = original + enhancement
        enhanced_hidden_states = original_hidden_states + alpha * combined
        
        # Return enhanced outputs while preserving original structure
        base_outputs.last_hidden_state = enhanced_hidden_states
        
        # Store enhancement info for analysis
        if not hasattr(base_outputs, 'enhancement_info'):
            base_outputs.enhancement_info = {}
        base_outputs.enhancement_info.update({
            'dtat_blend_factor': alpha.item(),
            'reasoning_steps': reasoning_info
        })
        
        return base_outputs
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Delegate to base model for generation
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def _reorder_cache(self, *args, **kwargs):
        # Delegate to base model for cache reordering
        return self.base_model._reorder_cache(*args, **kwargs)
    
    @property
    def device(self):
        return self.base_model.device
