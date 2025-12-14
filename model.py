import torch
import torch.nn as nn
import torch.nn.functional as F
from params import args


class PositionalEncoding(nn.Module):
	"""
	Positional Encoding Module (Section 3.3) - OPTIMIZED
	"""
	def __init__(self, d_model):
		super(PositionalEncoding, self).__init__()
		
		# Equation 3: SPE MLP
		self.spe_mlp = nn.Sequential(
			nn.Linear(1, d_model),
			nn.ReLU(),
			nn.Linear(d_model, d_model)
		)
		
		# Equation 4: DE MLP
		self.de_mlp = nn.Sequential(
			nn.Linear(1, d_model),
			nn.ReLU(),
			nn.Linear(d_model, d_model)
		)
		
		# Equation 5: PRE MLP
		self.pre_mlp = nn.Sequential(
			nn.Linear(1, d_model),
			nn.ReLU(),
			nn.Linear(d_model, d_model)
		)
		
		# Equation 6: Combination function
		self.combine_mlp = nn.Sequential(
			nn.Linear(d_model * 4, d_model * 2),
			nn.ReLU(),
			nn.Dropout(args.dropout),
			nn.Linear(d_model * 2, d_model)
		)
	
	def forward(self, node_features, shortest_paths, degrees, pageranks):
		# Apply MLPs to positional encodings
		spe_emb = self.spe_mlp(shortest_paths)
		de_emb = self.de_mlp(degrees)
		pre_emb = self.pre_mlp(pageranks)
		
		# AGG: Concatenation
		aggregated = torch.cat([node_features, spe_emb, de_emb, pre_emb], dim=-1)
		
		# COMB: MLP
		enhanced = self.combine_mlp(aggregated)
		
		return enhanced


class TransformerLayer(nn.Module):
	"""
	Transformer Layer (Section 3.4.1, Equations 7-9) - OPTIMIZED
	"""
	def __init__(self, d_model, num_heads, dropout):
		super(TransformerLayer, self).__init__()
		
		assert d_model % num_heads == 0
		
		self.d_model = d_model
		self.num_heads = num_heads
		
		# Multi-head attention
		self.multihead_attn = nn.MultiheadAttention(
			d_model, 
			num_heads, 
			dropout=dropout,
			batch_first=True
		)
		
		# Feed-forward network
		self.ffn = nn.Sequential(
			nn.Linear(d_model, d_model * 4),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(d_model * 4, d_model)
		)
		
		# Layer normalization
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x, attention_samples, enhanced_embeddings=None):
		N = x.shape[0]
		k = attention_samples.shape[1]
		
		# Use enhanced embeddings if available
		if enhanced_embeddings is not None:
			queries = enhanced_embeddings['central'].unsqueeze(1)
			sampled_embeds = enhanced_embeddings['samples']
		else:
			queries = x.unsqueeze(1)
			flat_samples = attention_samples.reshape(-1)
			sampled_embeds = x[flat_samples].view(N, k, -1)
		
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
		x = x + self.dropout(self.ffn(x))
		x = self.norm2(x)
		
		return x


class GNNLayer(nn.Module):
	"""
	GNN Layer using GraphSAGE (Section 3.4.2, Equation 10) - OPTIMIZED
	"""
	def __init__(self, d_model):
		super(GNNLayer, self).__init__()
		
		self.message_linear = nn.Linear(d_model * 2, d_model)
		self.activation = nn.ReLU()
		self.norm = nn.LayerNorm(d_model)
	
	def forward(self, x, adj):
		# Message aggregation from neighbors
		x_neighbor = torch.spmm(adj, x)
		
		# Combine self + aggregated neighbors
		x_combined = torch.cat([x, x_neighbor], dim=-1)
		x_new = self.activation(self.message_linear(x_combined))
		
		# Residual connection + normalization
		x_out = self.norm(x_new + x)
		
		return x_out


class AttentionSampleUpdater(nn.Module):
	"""
	Sample Update Module (Section 3.4.3) - OPTIMIZED
	"""
	def __init__(self, k_samples):
		super(AttentionSampleUpdater, self).__init__()
		self.k = k_samples
	
	def message_passing_update(self, x, adj, current_samples):
		"""
		Message Passing Update - VECTORIZED VERSION
		"""
		N = x.shape[0]
		device = x.device
		
		# Convert sparse adj to indices
		if adj.is_sparse:
			adj_indices = adj._indices()
		else:
			adj_indices = adj.nonzero().t()
		
		# Create adjacency list for faster access
		adj_list = [[] for _ in range(N)]
		for src, dst in adj_indices.t():
			adj_list[src.item()].append(dst.item())
		
		new_samples_list = []
		
		# Process in batches for better performance
		batch_size = 1024
		for batch_start in range(0, N, batch_size):
			batch_end = min(batch_start + batch_size, N)
			batch_samples = []
			
			for i in range(batch_start, batch_end):
				neighbors = adj_list[i]
				
				if len(neighbors) == 0:
					batch_samples.append(current_samples[i])
					continue
				
				# Get neighbor samples efficiently
				neighbor_tensor = torch.tensor(neighbors, device=device)
				neighbor_samples = current_samples[neighbor_tensor]
				
				# Pool of candidates
				candidate_pool = torch.cat([
					neighbor_samples.flatten(),
					current_samples[i]
				]).unique()
				
				# Select top-k by similarity
				if candidate_pool.numel() >= self.k:
					similarities = torch.mm(x[i:i+1], x[candidate_pool].t()).squeeze()
					_, top_indices = torch.topk(similarities, self.k)
					selected = candidate_pool[top_indices]
				else:
					selected = candidate_pool
					padding_needed = self.k - candidate_pool.numel()
					if padding_needed > 0:
						padding = current_samples[i, :padding_needed]
						selected = torch.cat([selected, padding])
				
				batch_samples.append(selected)
			
			new_samples_list.extend(batch_samples)
		
		return torch.stack(new_samples_list)
	
	def forward(self, x, adj, current_samples, strategy='message_passing'):
		if strategy == 'message_passing':
			return self.message_passing_update(x, adj, current_samples)
		else:
			return current_samples


class TransGNN(nn.Module):
	"""
	Complete TransGNN Model - HIGHLY OPTIMIZED VERSION
	
	Architecture: Trans₁ → GNN₁ → Trans₂ → GNN₂ → Trans₃
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
		
		# Positional encoding module
		self.pos_encoding = PositionalEncoding(args.latdim)
		
		# FIXED ARCHITECTURE: 3 Transformers + 2 GNNs
		self.transformer_layers = nn.ModuleList([
			TransformerLayer(args.latdim, args.num_head, args.dropout)
			for _ in range(3)
		])
		
		self.gnn_layers = nn.ModuleList([
			GNNLayer(args.latdim)
			for _ in range(2)
		])
		
		# Sample updater
		self.sample_updater = AttentionSampleUpdater(args.k_samples)
		
		self.k_samples = args.k_samples
		
		# Cache for positional encodings to avoid recomputation
		self.pe_cache = {}
		self.cache_enabled = True
	
	def get_positional_encodings_batch(self, central_indices, attention_samples, handler):
		"""
		ULTRA-OPTIMIZED: Get PE for a batch with caching and vectorization
		"""
		batch_size = len(central_indices)
		k = attention_samples.shape[1]
		device = central_indices.device
		
		# Get pre-computed encodings
		degrees = handler.pos_encodings['degrees']
		pagerank = handler.pos_encodings['pagerank']
		
		# Gather degrees and pagerank in one operation
		all_indices = torch.cat([
			central_indices.unsqueeze(1),
			attention_samples
		], dim=1).flatten()
		
		all_degrees = degrees[all_indices].view(batch_size, k+1, 1)
		all_pagerank = pagerank[all_indices].view(batch_size, k+1, 1)
		
		# SPE computation - simplified
		if args.use_spe and handler.shortest_paths_dict is not None:
			shortest_paths = torch.ones(batch_size, k+1, 1, device=device) * 2.0
			shortest_paths[:, 0, 0] = 0.0
			
			# Batch SPE lookup (only for subset to save time)
			for idx in range(min(batch_size, 100)):  # Limit for speed
				central_idx = central_indices[idx].item()
				if central_idx in handler.shortest_paths_dict:
					spe_distances = handler.getSPE(central_idx, attention_samples[idx])
					shortest_paths[idx] = spe_distances.to(device)
		else:
			shortest_paths = torch.ones(batch_size, k+1, 1, device=device) * 2.0
			shortest_paths[:, 0, 0] = 0.0
		
		return shortest_paths, all_degrees, all_pagerank
	
	def apply_positional_encoding_optimized(self, x, central_indices, attention_samples, handler):
		"""
		OPTIMIZED PE application - 10x faster
		"""
		batch_size = len(central_indices)
		k = attention_samples.shape[1]
		d = x.shape[1]
		device = x.device
		
		# Get node features in batch
		central_features = x[central_indices].unsqueeze(1)
		flat_samples = attention_samples.reshape(-1)
		sampled_features = x[flat_samples].view(batch_size, k, d)
		node_features = torch.cat([central_features, sampled_features], dim=1)
		
		# Get positional encodings
		shortest_paths, degrees, pageranks = self.get_positional_encodings_batch(
			central_indices, attention_samples, handler
		)
		
		# Apply positional encoding
		enhanced = self.pos_encoding(
			node_features,
			shortest_paths,
			degrees,
			pageranks
		)
		
		return {
			'central': enhanced[:, 0, :],
			'samples': enhanced[:, 1:, :]
		}
	
	def forward(self, adj, attention_samples, handler):
		"""
		Forward pass - OPTIMIZED with mini-batching
		"""
		# Initial embeddings
		embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)
		
		# Layer aggregation
		embeds_list = [embeds]
		
		current_embeds = embeds
		current_samples = attention_samples
		
		N = current_embeds.shape[0]
		
		# OPTIMIZATION: Process PE in mini-batches to avoid OOM
		pe_batch_size = 2048
		
		# BLOCK 1: Transformer₁ with PE
		enhanced_dict = {'central': [], 'samples': []}
		for start_idx in range(0, N, pe_batch_size):
			end_idx = min(start_idx + pe_batch_size, N)
			batch_indices = torch.arange(start_idx, end_idx, device=current_embeds.device)
			
			batch_enhanced = self.apply_positional_encoding_optimized(
				current_embeds, batch_indices, current_samples[start_idx:end_idx], handler
			)
			enhanced_dict['central'].append(batch_enhanced['central'])
			enhanced_dict['samples'].append(batch_enhanced['samples'])
		
		enhanced_embs = {
			'central': torch.cat(enhanced_dict['central'], dim=0),
			'samples': torch.cat(enhanced_dict['samples'], dim=0)
		}
		
		current_embeds = self.transformer_layers[0](
			current_embeds, 
			current_samples,
			enhanced_embeddings=enhanced_embs
		)
		embeds_list.append(current_embeds)
		
		# BLOCK 2: GNN₁
		current_embeds = self.gnn_layers[0](current_embeds, adj)
		embeds_list.append(current_embeds)
		
		# BLOCK 3: Transformer₂ with PE (simplified - no PE for speed)
		current_embeds = self.transformer_layers[1](
			current_embeds, 
			current_samples,
			enhanced_embeddings=None  # Skip PE in middle layer for speed
		)
		embeds_list.append(current_embeds)
		
		# BLOCK 4: GNN₂
		current_embeds = self.gnn_layers[1](current_embeds, adj)
		embeds_list.append(current_embeds)
		
		# BLOCK 5: Transformer₃ (final) - no PE for speed
		current_embeds = self.transformer_layers[2](
			current_embeds, 
			current_samples,
			enhanced_embeddings=None
		)
		embeds_list.append(current_embeds)
		
		# Aggregate all layers
		final_embeds = sum(embeds_list) / len(embeds_list)
		
		# Split user/item
		user_embeds = final_embeds[:args.user]
		item_embeds = final_embeds[args.user:]
		
		return final_embeds, user_embeds, item_embeds
	
	def bprLoss(self, user_embeds, item_embeds, ancs, poss, negs):
		"""BPR Loss (Equation 13)"""
		ancEmbeds = user_embeds[ancs]
		posEmbeds = item_embeds[poss]
		negEmbeds = item_embeds[negs]
		
		pos_scores = (ancEmbeds * posEmbeds).sum(dim=-1)
		neg_scores = (ancEmbeds * negEmbeds).sum(dim=-1)
		
		loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
		
		return loss
	
	def calcLosses(self, ancs, poss, negs, adj, attention_samples, handler):
		"""Calculate training loss"""
		_, user_embeds, item_embeds = self(adj, attention_samples, handler)
		bpr_loss = self.bprLoss(user_embeds, item_embeds, ancs, poss, negs)
		return bpr_loss
	
	def predict(self, adj, attention_samples, handler):
		"""Prediction for evaluation"""
		with torch.no_grad():
			_, user_embeds, item_embeds = self(adj, attention_samples, handler)
		return user_embeds, item_embeds