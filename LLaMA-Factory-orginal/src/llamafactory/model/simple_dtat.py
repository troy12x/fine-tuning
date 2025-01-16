import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    @torch.jit.script
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
        
        # Gradient checkpointing for memory efficiency
        self.use_checkpoint = True
        
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
    
    @torch.jit.script
    def get_concept_embeddings(self, x, input_ids, level):
        """Memory-efficient concept embedding lookup"""
        B, T = input_ids.shape
        H = self.hidden_size
        
        # Use sparse operations for memory efficiency
        concept_embeds = F.embedding(
            input_ids,
            self.level_concepts[level],
            sparse=True
        )  # [B, T, H]
        
        return concept_embeds
    
    def forward(self, x, input_ids, attention_mask=None):
        """Memory-efficient forward pass with gradient checkpointing"""
        B, T, H = x.size()
        all_concepts = []
        level_x = x
        
        # Process in chunks for memory efficiency
        chunk_size = min(T, 512)  # Adjust based on GPU memory
        
        for level in range(self.num_levels):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # Process chunks
            level_chunks = []
            for i in range(0, T, chunk_size):
                end_idx = min(i + chunk_size, T)
                chunk_mask = attention_mask[:, i:end_idx] if attention_mask is not None else None
                
                # Get concepts for chunk
                chunk_concepts = self.get_concept_embeddings(
                    x[:, i:end_idx],
                    input_ids[:, i:end_idx],
                    level
                )  # [B, chunk_size, H]
                
                # Process chunk with gradient checkpointing
                if self.use_checkpoint and self.training:
                    chunk_features = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.concept_layers[level]['extractor']),
                        level_x[:, i:end_idx],
                        chunk_mask
                    )  # [B, chunk_size, H]
                else:
                    chunk_features = self.concept_layers[level]['extractor'](
                        level_x[:, i:end_idx],
                        src_key_padding_mask=chunk_mask
                    )  # [B, chunk_size, H]
                
                # Project and combine
                chunk_concepts = self.concept_layers[level]['projector'](chunk_concepts)  # [B, chunk_size, H]
                chunk_combined = self.concept_layers[level]['aggregator'](
                    torch.cat([chunk_features, chunk_concepts], dim=-1)  # [B, chunk_size, 2H]
                )  # [B, chunk_size, H]
                
                # Add residual and normalize
                chunk_combined = self.norm(chunk_combined + level_x[:, i:end_idx])  # [B, chunk_size, H]
                level_chunks.append(chunk_combined)
            
            # Combine chunks
            combined = torch.cat(level_chunks, dim=1)  # [B, T, H]
            all_concepts.append(combined)
            
            # Cross-level attention
            if level < self.num_levels - 1:
                # Reshape for attention
                q = combined.view(B, T, H)  # [B, T, H]
                k = combined.view(B, T, H)  # [B, T, H]
                v = combined.view(B, T, H)  # [B, T, H]
                
                # Apply attention
                level_x, _ = self.cross_level_attention[level](
                    q, k, v,
                    key_padding_mask=attention_mask
                )  # [B, T, H]
        
        return all_concepts  # List of [B, T, H] tensors

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
    
    @torch.jit.script
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
    
    @torch.jit.script
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
    
    @torch.jit.script
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
    
    @torch.jit.script
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
        
        # Output integration
        self.output = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, input_ids, attention_mask=None):
        # 1. Process semantics
        semantic_state, semantic_info = self.semantic(
            x, input_ids, attention_mask
        )
        
        # 2. Compute temperatures
        temps = self.temperature(torch.cat([
            x, semantic_state
        ], dim=-1))
        
        # 3. Initialize reasoning
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
            
            # Update state
            state = state + best_step
            
            # Store progress
            all_states.append(state)
            all_validities.append(validities)
            
            # Check if reasoning is complete
            if validities.mean() > min_validity:
                break
        
        # 5. Combine reasoning steps
        reasoning_states = torch.stack(all_states, dim=1)
        reasoning_validities = torch.stack(all_validities, dim=1)
        
        # Weight by validities
        weights = F.softmax(reasoning_validities, dim=1).unsqueeze(-1)
        reasoned = (reasoning_states * weights).sum(1)
        
        # Final output
        output = self.output(torch.cat([x, reasoned], dim=-1))
        
        return output, {
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        
        # Reasoning layers with CoT
        self.reasoning_layers = nn.ModuleList([
            SimpleReasoningLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = True
        
        # Initialize weights
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"SimpleDTAT initialized with {n_params:,} parameters")

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
        
        # Store intermediate thoughts
        all_thoughts = []
        
        for layer in self.reasoning_layers:
            if self.gradient_checkpointing and self.training:
                x, thoughts = torch.utils.checkpoint.checkpoint(layer, x, input_ids, attention_mask)
            else:
                x, thoughts = layer(x, input_ids, attention_mask)
            all_thoughts.append(thoughts)
        
        x = self.ln_f(x)
        return x, all_thoughts  # Return both output and reasoning steps

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
