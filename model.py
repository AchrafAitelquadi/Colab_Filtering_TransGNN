import torch
import torch.nn as nn
import torch.nn.functional as F
from params import args


class PositionalEncoding(nn.Module):
	"""
	Positional Encoding Module (Section 3.3)
	Combines SPE, DE, and PRE using MLPs
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
		
		# Equation 6: Combination function COMB(AGG(...))
		# AGG is concatenation, COMB is MLP
		self.combine_mlp = nn.Sequential(
			nn.Linear(d_model * 4, d_model * 2),
			nn.ReLU(),
			nn.Dropout(args.dropout),
			nn.Linear(d_model * 2, d_model)
		)
	
	def forward(self, node_features, shortest_paths, degrees, pageranks):
		"""
		Equation 6 from paper:
		h_i = COMB(AGG(x_i, SPE(v_i, v_i), DE(v_i), PRE(v_i)))
		h_j = COMB(AGG(x_j, SPE(v_i, v_j), DE(v_j), PRE(v_j))) for v_j in Smp(v_i)
		
		Args:
			node_features: [N, k+1, d] - raw features of [central, samples]
			shortest_paths: [N, k+1, 1] - SPE values
			degrees: [N, k+1, 1] - DE values
			pageranks: [N, k+1, 1] - PRE values
		
		Returns:
			enhanced_features: [N, k+1, d]
		"""
		# Apply MLPs to positional encodings
		spe_emb = self.spe_mlp(shortest_paths)  # [N, k+1, d]
		de_emb = self.de_mlp(degrees)           # [N, k+1, d]
		pre_emb = self.pre_mlp(pageranks)       # [N, k+1, d]
		
		# AGG: Concatenation
		aggregated = torch.cat([node_features, spe_emb, de_emb, pre_emb], dim=-1)
		
		# COMB: MLP
		enhanced = self.combine_mlp(aggregated)
		
		return enhanced


class TransformerLayer(nn.Module):
	"""
	Transformer Layer (Section 3.4.1, Equations 7-9)
	"""
	def __init__(self, d_model, num_heads, dropout):
		super(TransformerLayer, self).__init__()
		
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
		
		self.d_model = d_model
		self.num_heads = num_heads
		
		# Multi-head attention (Equation 9)
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
		"""
		Equations 7-9 avec PE enrichis
		
		Args:
			x: [N, d] - node embeddings (fallback si enhanced_embeddings=None)
			attention_samples: [N, k] - sampled node indices
			enhanced_embeddings: dict ou None
				Si fourni : {'central': [N, d], 'samples': [N, k, d]}
				Sinon : utilise x directement
		
		Returns:
			x: [N, d] - updated embeddings
		"""
		N = x.shape[0]
		k = attention_samples.shape[1]
		
		# ✅ CORRECTION : Utiliser les embeddings enrichis si disponibles
		if enhanced_embeddings is not None:
			# Cas 1 : PE appliqué, utiliser les embeddings enrichis
			queries = enhanced_embeddings['central'].unsqueeze(1)  # [N, 1, d]
			sampled_embeds = enhanced_embeddings['samples']        # [N, k, d]
		else:
			# Cas 2 : Pas de PE (fallback), comportement original
			queries = x.unsqueeze(1)  # [N, 1, d]
			flat_samples = attention_samples.view(-1)
			sampled_embeds = x[flat_samples].view(N, k, -1)
		
		# Multi-head attention (reste identique)
		attn_output, attn_weights = self.multihead_attn(
			queries,           # Q: [N, 1, d]
			sampled_embeds,    # K: [N, k, d]
			sampled_embeds     # V: [N, k, d]
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
	GNN Layer using GraphSAGE (Section 3.4.2, Equation 10)
	h_M(v_i) = Message(h_k, for all v_k in N(v_i))
	h_i = Combine(h_i, h_M(v_i))
	"""
	def __init__(self, d_model):
		super(GNNLayer, self).__init__()
		
		# Message + Combine function
		self.message_linear = nn.Linear(d_model * 2, d_model)
		self.activation = nn.ReLU()
		self.norm = nn.LayerNorm(d_model)
	
	def forward(self, x, adj):
		"""
		Args:
			x: [N, d] - node embeddings
			adj: sparse adjacency matrix [N, N]
		
		Returns:
			x: [N, d] - updated embeddings
		"""
		# Message aggregation from neighbors
		x_neighbor = torch.spmm(adj, x)  # [N, d]
		
		# Combine self + aggregated neighbors (GraphSAGE style)
		x_combined = torch.cat([x, x_neighbor], dim=-1)  # [N, 2d]
		x_new = self.activation(self.message_linear(x_combined))  # [N, d]
		
		# Residual connection + normalization
		x_out = self.norm(x_new + x)
		
		return x_out


class AttentionSampleUpdater(nn.Module):
	"""
	Sample Update Module (Section 3.4.3) - VERSION OPTIMISÉE
	"""
	def __init__(self, k_samples):
		super(AttentionSampleUpdater, self).__init__()
		self.k = k_samples
	
	def message_passing_update(self, x, adj, current_samples):
		"""
		VERSION SIMPLIFIÉE selon Section 3.4.3, Equation 12
		Attn_Msg(v_i) = Union of Smp(v_j) for all v_j in N(v_i)
		"""
		N = x.shape[0]
		device = x.device
		
		# Convert sparse adj to indices
		if adj.is_sparse:
			adj_indices = adj._indices()  # [2, num_edges]
		else:
			adj_indices = adj.nonzero().t()
		
		new_samples_list = []
		
		for i in range(N):
			# Trouver les voisins de i
			neighbor_mask = adj_indices[0] == i
			neighbors = adj_indices[1][neighbor_mask]
			
			if neighbors.numel() == 0:
				# Pas de voisins, garder samples actuels
				new_samples_list.append(current_samples[i])
				continue
			
			# Récupérer samples des voisins : Union of Smp(v_j)
			neighbor_samples = current_samples[neighbors]  # [num_neighbors, k]
			
			# Pool de candidats : samples des voisins + samples actuels
			candidate_pool = torch.cat([
				neighbor_samples.flatten(),
				current_samples[i]
			]).unique()
			
			# Sélectionner top-k par similarité
			if candidate_pool.numel() >= self.k:
				similarities = torch.mm(x[i:i+1], x[candidate_pool].t()).squeeze()
				_, top_indices = torch.topk(similarities, self.k)
				selected = candidate_pool[top_indices]
			else:
				# Padding si nécessaire
				selected = candidate_pool
				padding_needed = self.k - candidate_pool.numel()
				if padding_needed > 0:
					padding = current_samples[i, :padding_needed]
					selected = torch.cat([selected, padding])
			
			new_samples_list.append(selected)
		
		return torch.stack(new_samples_list)
	
	def random_walk_update(self, x, adj, current_samples, walk_length=5):
		"""
		VERSION VECTORISÉE du Random Walk
		"""
		N = x.shape[0]
		device = x.device
		
		if adj.is_sparse:
			adj_dense = adj.to_dense()
		else:
			adj_dense = adj
		
		new_samples_list = []
		
		# Batch processing
		batch_size = 512
		for start_idx in range(0, N, batch_size):
			end_idx = min(start_idx + batch_size, N)
			batch_size_actual = end_idx - start_idx
			
			explored_sets = [set() for _ in range(batch_size_actual)]
			
			# Pour chaque nœud du batch, faire des walks depuis ses samples
			for local_idx in range(batch_size_actual):
				i = start_idx + local_idx
				
				# Partir de chaque sample
				for sample_node in current_samples[i]:
					current_node = sample_node.item()
					
					for _ in range(walk_length):
						explored_sets[local_idx].add(current_node)
						
						# Voisins
						neighbors = adj_dense[current_node].nonzero(as_tuple=True)[0]
						
						if neighbors.numel() == 0:
							break
						
						# Transition probabilities (Equation 11)
						neighbor_embeds = x[neighbors]
						current_embed = x[current_node:current_node+1]
						
						similarities = torch.mm(current_embed, neighbor_embeds.t()).squeeze()
						
						# Éviter les valeurs négatives
						similarities = F.relu(similarities) + 1e-8
						probs = similarities / similarities.sum()
						
						# Sample next node
						try:
							next_idx = torch.multinomial(probs, 1).item()
							current_node = neighbors[next_idx].item()
						except:
							break
			
			# Sélectionner top-k depuis explored
			for local_idx in range(batch_size_actual):
				i = start_idx + local_idx
				explored_list = list(explored_sets[local_idx])
				
				if len(explored_list) >= self.k:
					explored_tensor = torch.tensor(explored_list, device=device)
					similarities = torch.mm(x[i:i+1], x[explored_tensor].t()).squeeze()
					_, top_indices = torch.topk(similarities, self.k)
					selected = explored_tensor[top_indices]
				else:
					selected = torch.tensor(explored_list, device=device)
					padding = current_samples[i, :self.k - len(selected)]
					selected = torch.cat([selected, padding])
				
				new_samples_list.append(selected)
		
		return torch.stack(new_samples_list)
	
	def forward(self, x, adj, current_samples, strategy='message_passing'):
		"""Update attention samples"""
		if strategy == 'message_passing':
			return self.message_passing_update(x, adj, current_samples)
		elif strategy == 'random_walk':
			return self.random_walk_update(x, adj, current_samples, args.random_walk_length)
		else:
			return current_samples
		

class TransGNNBlock(nn.Module):
	"""
	Complete TransGNN Block (Section 3.4)
	Transformer Layer -> GNN Layer -> Sample Update
	"""
	def __init__(self, d_model, num_heads, dropout, k_samples):
		super(TransGNNBlock, self).__init__()
		
		self.transformer = TransformerLayer(d_model, num_heads, dropout)
		self.gnn = GNNLayer(d_model)
		self.sample_updater = AttentionSampleUpdater(k_samples)
	
	def forward(self, x, adj, attention_samples, update_samples=True):
		"""
		Args:
			x: [N, d] - node embeddings
			adj: adjacency matrix
			attention_samples: [N, k] - attention samples
			update_samples: whether to update samples
		
		Returns:
			x: [N, d] - updated embeddings
			attention_samples: [N, k] - possibly updated samples
		"""
		# Transformer layer (Section 3.4.1)
		x = self.transformer(x, attention_samples)
		
		# GNN layer (Section 3.4.2)
		x = self.gnn(x, adj)
		
		# Sample update (Section 3.4.3)
		if update_samples and args.update_strategy != 'none':
			attention_samples = self.sample_updater(
				x, adj, attention_samples, strategy=args.update_strategy
			)
		
		return x, attention_samples


class TransGNN(nn.Module):
	"""
	Complete TransGNN Model - ARCHITECTURE EXACTE DE L'ARTICLE
	
	Section 4.1.4: "We use three Transformer layers with two GNN layers 
	sandwiched between them"
	
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
		
		# Positional encoding module (Section 3.3)
		self.pos_encoding = PositionalEncoding(args.latdim)
		
		# ✅ ARCHITECTURE FIXE : 3 Transformers + 2 GNNs
		self.transformer_layers = nn.ModuleList([
			TransformerLayer(args.latdim, args.num_head, args.dropout)
			for _ in range(3)  # EXACTEMENT 3
		])
		
		self.gnn_layers = nn.ModuleList([
			GNNLayer(args.latdim)
			for _ in range(2)  # EXACTEMENT 2
		])
		
		# Sample updater
		self.sample_updater = AttentionSampleUpdater(args.k_samples)
		
		self.k_samples = args.k_samples
	
	def apply_positional_encoding(self, x, central_indices, attention_samples, handler):
		"""
		Applique PE pour un batch de nœuds centraux ET leurs samples
		
		Args:
			x: [N, d] - tous les embeddings
			central_indices: [batch] - indices des nœuds centraux
			attention_samples: [batch, k] - samples pour ces nœuds
			handler: DataHandler
		
		Returns:
			enhanced_embeddings: dict {
				'central': [batch, d],
				'samples': [batch, k, d]
			}
		"""
		batch_size = len(central_indices)
		device = x.device
		
		# Préparer features [batch, k+1, d]
		all_features = []
		all_spe = []
		all_de = []
		all_pre = []
		
		degrees = handler.pos_encodings['degrees']
		pagerank = handler.pos_encodings['pagerank']
		
		for idx, central_node in enumerate(central_indices):
			# Central node
			central_feat = x[central_node:central_node+1]  # [1, d]
			
			# Sampled nodes
			sampled_feats = x[attention_samples[idx]]  # [k, d]
			
			# Combine: [1+k, d]
			node_feats = torch.cat([central_feat, sampled_feats], dim=0)
			
			# SPE
			spe = handler.getSPE(central_node.item(), attention_samples[idx]).to(device)
			
			# DE et PRE
			de = torch.cat([degrees[central_node:central_node+1], 
							degrees[attention_samples[idx]]], dim=0)
			pre = torch.cat([pagerank[central_node:central_node+1], 
							pagerank[attention_samples[idx]]], dim=0)
			
			all_features.append(node_feats)
			all_spe.append(spe)
			all_de.append(de)
			all_pre.append(pre)
		
		# Stack
		node_features = torch.stack(all_features)  # [batch, k+1, d]
		shortest_paths = torch.stack(all_spe)      # [batch, k+1, 1]
		degrees_gathered = torch.stack(all_de)     # [batch, k+1, 1]
		pageranks_gathered = torch.stack(all_pre)  # [batch, k+1, 1]
		
		# Apply PE
		enhanced = self.pos_encoding(
			node_features, shortest_paths, degrees_gathered, pageranks_gathered
		)  # [batch, k+1, d]
		
		return {
			'central': enhanced[:, 0, :],      # [batch, d]
			'samples': enhanced[:, 1:, :]      # [batch, k, d]
		}
	
	def forward(self, adj, attention_samples, handler):
		"""
		Forward pass - ARCHITECTURE EXACTE avec PE corrigé
		Trans₁ → GNN₁ → Trans₂ → GNN₂ → Trans₃
		"""
		# Initial embeddings
		embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)
		
		# Layer aggregation
		embeds_list = [embeds]
		
		current_embeds = embeds
		current_samples = attention_samples
		
		N = current_embeds.shape[0]
		central_indices = torch.arange(N, device=current_embeds.device)
		
		# ========================================================================
		# BLOCK 1: Transformer₁ avec PE
		# ========================================================================
		enhanced_embs = self.apply_positional_encoding(
			current_embeds, central_indices, current_samples, handler
		)
		
		current_embeds = self.transformer_layers[0](
			current_embeds, 
			current_samples,
			enhanced_embeddings=enhanced_embs
		)
		embeds_list.append(current_embeds)
		
		# Update samples (optionnel selon args)
		if args.update_every_block and args.update_strategy != 'none':
			current_samples = self.sample_updater(
				current_embeds, adj, current_samples, strategy=args.update_strategy
			)
		
		# ========================================================================
		# BLOCK 2: GNN₁
		# ========================================================================
		current_embeds = self.gnn_layers[0](current_embeds, adj)
		embeds_list.append(current_embeds)
		
		# ========================================================================
		# BLOCK 3: Transformer₂ avec PE
		# ========================================================================
		enhanced_embs = self.apply_positional_encoding(
			current_embeds, central_indices, current_samples, handler
		)
		
		current_embeds = self.transformer_layers[1](
			current_embeds, 
			current_samples,
			enhanced_embeddings=enhanced_embs
		)
		embeds_list.append(current_embeds)
		
		if args.update_every_block and args.update_strategy != 'none':
			current_samples = self.sample_updater(
				current_embeds, adj, current_samples, strategy=args.update_strategy
			)
		
		# ========================================================================
		# BLOCK 4: GNN₂
		# ========================================================================
		current_embeds = self.gnn_layers[1](current_embeds, adj)
		embeds_list.append(current_embeds)
		
		# ========================================================================
		# BLOCK 5: Transformer₃ (final) avec PE
		# ========================================================================
		enhanced_embs = self.apply_positional_encoding(
			current_embeds, central_indices, current_samples, handler
		)
		
		current_embeds = self.transformer_layers[2](
			current_embeds, 
			current_samples,
			enhanced_embeddings=enhanced_embs
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