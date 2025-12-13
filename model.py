import torch
import torch.nn as nn
import torch.nn.functional as F
from params import args
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import networkx as nx


class PositionalEncoding(nn.Module):
    """
    Module pour les trois types de positional encoding:
    1. Shortest Path Hop (SPE)
    2. Degree-based (DE)
    3. PageRank-based (PRE)
    """
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        
        # MLPs pour chaque type d'encoding
        self.spe_mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.de_mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.pre_mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # MLP de combinaison
        self.combine_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, x, spe, de, pre):
        """
        x: raw node features [batch, d_model]
        spe: shortest path encoding [batch, 1]
        de: degree encoding [batch, 1]
        pre: pagerank encoding [batch, 1]
        """
        spe_emb = self.spe_mlp(spe)
        de_emb = self.de_mlp(de)
        pre_emb = self.pre_mlp(pre)
        
        # Concatenate all encodings
        combined = torch.cat([x, spe_emb, de_emb, pre_emb], dim=-1)
        output = self.combine_mlp(combined)
        
        return output


class TransformerLayer(nn.Module):
    """
    Transformer layer avec multi-head attention sur les samples
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
        k = attention_samples.shape[1]
        
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
    GNN layer - utilise GraphSAGE comme backbone
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
    Modèle TransGNN complet
    """
    def __init__(self):
        super(TransGNN, self).__init__()
        
        # Embeddings initiaux
        self.user_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(args.user, args.latdim))
        )
        self.item_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(args.item, args.latdim))
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(args.latdim)
        
        # TransGNN blocks (alternating Transformer and GNN)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(args.latdim, args.num_head, args.dropout)
            for _ in range(args.block_num + 1)  # block_num + 1 transformer layers
        ])
        
        self.gnn_layers = nn.ModuleList([
            GNNLayer(args.latdim)
            for _ in range(args.block_num)  # block_num GNN layers
        ])
        
        # Attention sampling parameters
        self.k_samples = args.k_samples if hasattr(args, 'k_samples') else 20
        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.5
    
    def forward(self, adj, attention_samples=None, pos_encodings=None):
        """
        Forward pass complet
        
        Args:
            adj: adjacency matrix (sparse tensor)
            attention_samples: [N, k] pre-computed attention samples
            pos_encodings: tuple of (degrees, pagerank) tensors
        """
        device = self.user_embedding.device
        N = args.user + args.item
        
        # Embeddings initiaux
        embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        embeds_list = [embeds]
        
        # Get positional encodings
        if pos_encodings is None:
            # Create dummy encodings if not provided
            degrees = torch.zeros(N, 1).to(device)
            pagerank = torch.ones(N, 1).to(device) / N
        else:
            degrees, pagerank = pos_encodings
        
        # If attention samples not provided, use neighbors (simplified)
        if attention_samples is None:
            # Simple fallback: sample random nodes
            attention_samples = torch.randint(0, N, (N, self.k_samples)).to(device)
        
        # TransGNN blocks: alternating Transformer and GNN
        current_embeds = embeds
        
        for i in range(args.block_num):
            # 1. Transformer layer - NO POSITIONAL ENCODING INSIDE LOOP
            # (Positional encoding is expensive, apply once at the end)
            current_embeds = self.transformer_layers[i](current_embeds, attention_samples)
            
            # 2. GNN layer
            current_embeds = self.gnn_layers[i](current_embeds, adj)
            
            # 3. Residual connection
            current_embeds = current_embeds + embeds_list[-1]
            
            embeds_list.append(current_embeds)
        
        # Final transformer layer
        current_embeds = self.transformer_layers[-1](current_embeds, attention_samples)
        
        # Apply positional encoding ONCE at the end (more efficient)
        self_shortest_path = torch.zeros(N, 1).to(device)
        current_embeds = self.pos_encoding(
            current_embeds, 
            self_shortest_path, 
            degrees, 
            pagerank
        )
        
        embeds_list.append(current_embeds)
        
        # Aggregate all layers
        final_embeds = sum(embeds_list) / len(embeds_list)
        
        user_embeds = final_embeds[:args.user]
        item_embeds = final_embeds[args.user:]
        
        return final_embeds, user_embeds, item_embeds
    
    def bprLoss(self, user_embeds, item_embeds, ancs, poss, negs):
        """
        BPR Loss pour ranking
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
        
        Args:
            ancs: anchor user indices
            poss: positive item indices
            negs: negative item indices
            adj: adjacency matrix
            attention_samples: pre-computed attention samples
            pos_encodings: positional encodings
        """
        embeds, user_embeds, item_embeds = self(adj, attention_samples, pos_encodings)
        
        bpr_loss = self.bprLoss(user_embeds, item_embeds, ancs, poss, negs)
        
        return bpr_loss
    
    def predict(self, adj, attention_samples=None, pos_encodings=None):
        """
        Prediction pour test
        
        Args:
            adj: adjacency matrix
            attention_samples: pre-computed attention samples
            pos_encodings: positional encodings
        """
        with torch.no_grad():
            embeds, user_embeds, item_embeds = self(adj, attention_samples, pos_encodings)
        return user_embeds, item_embeds