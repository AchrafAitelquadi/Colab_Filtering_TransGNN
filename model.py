import torch
import torch.nn as nn
import torch.nn.functional as F
from params import args


class PositionalEncoding(nn.Module):
    """
    Simplified positional encoding that applies BEFORE TransGNN blocks
    According to paper Section 3.3 and Equation 6
    """
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        
        # Single MLP for each encoding type (simplified)
        self.spe_mlp = nn.Linear(1, d_model)
        self.de_mlp = nn.Linear(1, d_model)
        self.pre_mlp = nn.Linear(1, d_model)
        
        # Simple combination - element-wise addition instead of concat+MLP
        # This drastically reduces parameters
    
    def forward(self, x, spe, de, pre):
        """
        x: raw node features [N, d_model]
        spe: shortest path encoding [N, 1] 
        de: degree encoding [N, 1]
        pre: pagerank encoding [N, 1]
        
        Returns: enhanced features [N, d_model]
        """
        # Project each encoding to d_model dimension
        spe_emb = self.spe_mlp(spe)  # [N, d_model]
        de_emb = self.de_mlp(de)      # [N, d_model]
        pre_emb = self.pre_mlp(pre)   # [N, d_model]
        
        # Combine via addition (much fewer parameters than concat+MLP)
        # According to paper Equation 6, these are aggregated
        output = x + spe_emb + de_emb + pre_emb
        
        return output


class TransformerLayer(nn.Module):
    """
    Transformer layer with multi-head attention on sampled nodes
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            d_model, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x, attention_samples):
        """
        x: [N, d_model] - all node embeddings
        attention_samples: [N, k] - indices of sampled nodes for each node
        """
        N = x.shape[0]
        
        # Prepare queries, keys, values
        queries = x.unsqueeze(1)  # [N, 1, d_model]
        
        # Gather sampled nodes for each central node
        sampled_embeds = x[attention_samples]  # [N, k, d_model]
        
        # Multi-head attention
        attn_output, _ = self.multihead_attn(
            queries, 
            sampled_embeds, 
            sampled_embeds
        )
        
        # Add & Norm
        x = x + self.dropout(attn_output.squeeze(1))
        x = self.norm1(x)
        
        # Feed-forward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x


class GNNLayer(nn.Module):
    """
    GNN layer - GraphSAGE style message passing
    """
    def __init__(self, d_model):
        super(GNNLayer, self).__init__()
        
        self.linear = nn.Linear(d_model * 2, d_model)
        self.activation = nn.ReLU()
    
    def forward(self, x, adj):
        """
        x: [N, d_model]
        adj: sparse adjacency matrix
        """
        # Message passing
        x_neighbor = torch.spmm(adj, x)
        
        # Combine with self
        x_combined = torch.cat([x, x_neighbor], dim=-1)
        x_out = self.activation(self.linear(x_combined))
        
        return x_out


class TransGNN(nn.Module):
    """
    CORRECTED TransGNN Model
    
    Key fixes:
    1. Positional encoding applied BEFORE TransGNN blocks
    2. Correct architecture: Trans → GNN → Trans → GNN → Trans
    3. Simplified positional encoding (fewer parameters)
    """
    def __init__(self):
        super(TransGNN, self).__init__()
        
        # Initial embeddings
        self.user_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(args.user, args.latdim))
        )
        self.item_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(args.item, args.latdim))
        )
        
        # Positional encoding (applied ONCE at the beginning)
        self.pos_encoding = PositionalEncoding(args.latdim)
        
        # Architecture: 3 Transformer layers, 2 GNN layers
        # Trans → GNN → Trans → GNN → Trans
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(args.latdim, args.num_head, args.dropout)
            for _ in range(3)  # Fixed: exactly 3 transformer layers
        ])
        
        self.gnn_layers = nn.ModuleList([
            GNNLayer(args.latdim)
            for _ in range(2)  # Fixed: exactly 2 GNN layers
        ])
        
        # Attention sampling parameters
        self.k_samples = args.k_samples if hasattr(args, 'k_samples') else 20
        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.5
    
    def forward(self, adj, attention_samples=None, pos_encodings=None):
        """
        CORRECTED Forward pass
        
        Architecture flow:
        1. Get initial embeddings
        2. Apply positional encoding ONCE
        3. Trans → GNN → Trans → GNN → Trans with residual connections
        4. Aggregate all layer outputs
        """
        device = self.user_embedding.device
        N = args.user + args.item
        
        # Step 1: Initial embeddings
        embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        
        # Step 2: Apply positional encoding BEFORE TransGNN blocks (CRITICAL FIX)
        if pos_encodings is None:
            degrees = torch.zeros(N, 1).to(device)
            pagerank = torch.ones(N, 1).to(device) / N
        else:
            degrees, pagerank = pos_encodings
        
        # Shortest path from node to itself is 0
        self_shortest_path = torch.zeros(N, 1).to(device)
        
        # APPLY POSITIONAL ENCODING HERE (not at the end!)
        embeds = self.pos_encoding(embeds, self_shortest_path, degrees, pagerank)
        
        # Store for residual connections and final aggregation
        embeds_list = [embeds]
        
        # Fallback for attention samples
        if attention_samples is None:
            attention_samples = torch.randint(0, N, (N, self.k_samples)).to(device)
        
        # Step 3: TransGNN blocks with correct architecture
        # Architecture: Trans → GNN → Trans → GNN → Trans
        current_embeds = embeds
        
        # First Transformer layer
        current_embeds = self.transformer_layers[0](current_embeds, attention_samples)
        current_embeds = current_embeds + embeds_list[-1]  # Residual
        embeds_list.append(current_embeds)
        
        # First GNN layer
        current_embeds = self.gnn_layers[0](current_embeds, adj)
        current_embeds = current_embeds + embeds_list[-1]  # Residual
        embeds_list.append(current_embeds)
        
        # Second Transformer layer
        current_embeds = self.transformer_layers[1](current_embeds, attention_samples)
        current_embeds = current_embeds + embeds_list[-1]  # Residual
        embeds_list.append(current_embeds)
        
        # Second GNN layer
        current_embeds = self.gnn_layers[1](current_embeds, adj)
        current_embeds = current_embeds + embeds_list[-1]  # Residual
        embeds_list.append(current_embeds)
        
        # Third (final) Transformer layer
        current_embeds = self.transformer_layers[2](current_embeds, attention_samples)
        current_embeds = current_embeds + embeds_list[-1]  # Residual
        embeds_list.append(current_embeds)
        
        # Step 4: Aggregate all layers (paper mentions this)
        final_embeds = sum(embeds_list) / len(embeds_list)
        
        # Split into user and item embeddings
        user_embeds = final_embeds[:args.user]
        item_embeds = final_embeds[args.user:]
        
        return final_embeds, user_embeds, item_embeds
    
    def bprLoss(self, user_embeds, item_embeds, ancs, poss, negs):
        """
        BPR Loss for ranking
        """
        ancEmbeds = user_embeds[ancs]
        posEmbeds = item_embeds[poss]
        negEmbeds = item_embeds[negs]
        
        pos_scores = (ancEmbeds * posEmbeds).sum(dim=-1)
        neg_scores = (ancEmbeds * negEmbeds).sum(dim=-1)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-6).mean()
        
        return bpr_loss
    
    def calcLosses(self, ancs, poss, negs, adj, attention_samples=None, pos_encodings=None):
        """
        Calculate losses
        """
        embeds, user_embeds, item_embeds = self(adj, attention_samples, pos_encodings)
        
        bpr_loss = self.bprLoss(user_embeds, item_embeds, ancs, poss, negs)
        
        return bpr_loss
    
    def predict(self, adj, attention_samples=None, pos_encodings=None):
        """
        Prediction for test
        """
        with torch.no_grad():
            embeds, user_embeds, item_embeds = self(adj, attention_samples, pos_encodings)
        return user_embeds, item_embeds


# Print parameter count for comparison
if __name__ == '__main__':
    print("=" * 60)
    print("TransGNN Model Architecture Summary")
    print("=" * 60)
    
    # Create dummy args
    class DummyArgs:
        user = 1000
        item = 1000
        latdim = 64
        num_head = 4
        dropout = 0.1
        k_samples = 20
        alpha = 0.5
    
    args = DummyArgs()
    
    model = TransGNN()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nBreakdown by component:")
    print(f"  User embeddings: {args.user * args.latdim:,}")
    print(f"  Item embeddings: {args.item * args.latdim:,}")
    
    pos_params = sum(p.numel() for p in model.pos_encoding.parameters())
    print(f"  Positional encoding: {pos_params:,}")
    
    trans_params = sum(sum(p.numel() for p in layer.parameters()) 
                       for layer in model.transformer_layers)
    print(f"  Transformer layers (3x): {trans_params:,}")
    
    gnn_params = sum(sum(p.numel() for p in layer.parameters()) 
                     for layer in model.gnn_layers)
    print(f"  GNN layers (2x): {gnn_params:,}")
    
    print("\n" + "=" * 60)
    print("Architecture: Trans → GNN → Trans → GNN → Trans")
    print("Positional encoding applied BEFORE blocks (not after)")
    print("=" * 60)