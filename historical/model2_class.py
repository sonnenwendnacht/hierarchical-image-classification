from torchvision.models import resnet18, ResNet18_Weights

class SoftGatedExpertModel(nn.Module):
    def __init__(self, num_super_classes, sub_map_df, hidden_dim=256):
        super().__init__()
        
        # 1. Backbone: ResNet18 (Pre-trained)
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        
        # 2. Gating Network
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_super_classes) 
        )
        
        # 3. Build Experts
        self.experts = nn.ModuleList()
        self.expert_indices = []
        
        unique_supers = sorted(sub_map_df['superclass_index'].unique())
        self.total_sub_classes = len(sub_map_df) 
        
        for super_idx in unique_supers:
            indices = sub_map_df[sub_map_df['superclass_index'] == super_idx].index.tolist()
            
            # --- FIX: Explicitly set dtype to torch.long ---
            self.expert_indices.append(torch.tensor(indices, dtype=torch.long))
            # ------------------------------------------------
            
            expert_head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, len(indices)) 
            )
            self.experts.append(expert_head)

    def forward(self, x):
        # 1. Extract Features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # 2. Gating
        gate_logits = self.gate(features) 
        gate_probs = F.softmax(gate_logits, dim=1)
        
        # 3. Expert Execution & Soft Aggregation
        final_sub_probs = torch.zeros(x.size(0), self.total_sub_classes, device=x.device)
        
        for i, expert in enumerate(self.experts):
            # Use safe indexing for gate probs (assuming sorted unique_supers aligns with 0,1,2...)
            gate_weight = gate_probs[:, i].unsqueeze(1)
            
            expert_logits = expert(features)
            expert_probs = F.softmax(expert_logits, dim=1)
            
            weighted_expert_probs = expert_probs * gate_weight
            
            indices = self.expert_indices[i].to(x.device)
            
            # This line crashed before; now it should work because indices are Long
            final_sub_probs[:, indices] += weighted_expert_probs
            
        return gate_logits, final_sub_probs