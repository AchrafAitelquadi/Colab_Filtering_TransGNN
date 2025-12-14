import pickle
import numpy as np
from scipy.sparse import coo_matrix
from params import args
import scipy.sparse as sp
from utils.timeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import networkx as nx
from collections import defaultdict

class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = 'data/yelp/'
		elif args.data == 'ml10m':
			predir = 'data/ml10m/'
		elif args.data == 'tmall':
			predir = 'data/tmall/'
		elif args.data == 'gowalla':
			predir = 'data/gowalla/'
		elif args.data == 'amazon-book':
			predir = 'data/amazon-book/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		
		# Positional encodings
		self.pos_encodings = None
		self.shortest_paths_dict = None  # Pour stockage efficace des SPE

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		"""Create normalized adjacency matrix for GNN"""
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		
		sparse_tensor = t.sparse_coo_tensor(idxs, vals, shape)
		return sparse_tensor.cuda() if t.cuda.is_available() else sparse_tensor

	def computeShortestPaths(self, adj_matrix):
		"""
		OPTIMIZED: Fast approximate SPE using sparse matrix powers
		Computing exact shortest paths for all pairs is O(N^3) - too slow!
		
		Instead: Use sparse matrix multiplication (adjacency powers) for k-hop neighbors
		This gives us approximate distances in O(k * E) time
		"""
		log('Computing Shortest Path Encoding (SPE) - FAST VERSION...', level='INFO')
		
		N = adj_matrix.shape[0]
		max_hops = min(args.max_spe_distance, 5)  # Limit to 5 hops for speed
		
		# Initialize distance matrix (sparse)
		import scipy.sparse as sp
		
		# Start with adjacency (1-hop connections)
		A = adj_matrix.copy()
		A = (A > 0).astype(np.float32)  # Binary
		
		# Distance dictionary: {source: {target: distance}}
		shortest_paths_dict = {}
		
		# Convert to LIL format for efficient updates
		dist_matrix = sp.lil_matrix((N, N), dtype=np.int8)
		dist_matrix.setdiag(0)  # Distance to self = 0
		
		# Mark 1-hop neighbors
		dist_matrix[A.nonzero()] = 1
		
		# Compute k-hop neighbors iteratively
		A_power = A.copy()
		
		for hop in range(2, max_hops + 1):
			if hop % 2 == 0:
				log(f'Computing {hop}-hop distances...', level='DEBUG', save=False, oneline=True)
			
			# A^k gives k-hop reachability
			A_power = A_power.dot(A)
			
			# Find NEW k-hop neighbors (not seen before)
			new_neighbors = (A_power > 0).multiply(dist_matrix == 0)
			
			# Set their distance to k
			dist_matrix[new_neighbors.nonzero()] = hop
			
			# Early stopping if no new neighbors
			if new_neighbors.nnz == 0:
				break
		
		print()  # New line
		
		# Sample nodes to store (for memory efficiency)
		if N > args.spe_sample_size:
			sample_nodes = np.random.choice(N, args.spe_sample_size, replace=False)
			log(f'Storing SPE for {len(sample_nodes)} sampled nodes', level='INFO')
		else:
			sample_nodes = range(N)
		
		# Convert to dictionary format (only for sampled nodes)
		dist_matrix = dist_matrix.tocsr()
		
		for source in sample_nodes:
			row = dist_matrix.getrow(source)
			# Only store non-zero distances (reachable nodes)
			targets = row.nonzero()[1]
			distances = row.data
			
			shortest_paths_dict[source] = {
				int(target): int(dist) for target, dist in zip(targets, distances)
			}
			shortest_paths_dict[source][source] = 0  # Self-distance
		
		log(f'SPE computed for {len(shortest_paths_dict)} nodes in {max_hops} hops', level='SUCCESS')
		log(f'Average neighbors per node: {sum(len(v) for v in shortest_paths_dict.values()) / len(shortest_paths_dict):.1f}', level='INFO')
		
		return shortest_paths_dict

	def computePositionalEncodings(self, adj_matrix):
		"""
		Compute all positional encodings (Section 3.3)
		- Shortest Path Encoding (SPE)
		- Degree Encoding (DE)
		- PageRank Encoding (PRE)
		"""
		log('Computing positional encodings...', level='INFO')
		
		N = adj_matrix.shape[0]
		device = t.device('cuda' if t.cuda.is_available() else 'cpu')
		
		# 1. Degree Encoding (Section 3.3.2)
		degrees = np.array(adj_matrix.sum(axis=1)).flatten()
		degrees_tensor = t.from_numpy(degrees).float().unsqueeze(1).to(device)
		
		# 2. PageRank Encoding (Section 3.3.3)
		# Fast approximation: degree-based
		total_degree = degrees.sum() + 1e-8
		pagerank_values = degrees / total_degree
		pagerank_tensor = t.from_numpy(pagerank_values).float().unsqueeze(1).to(device)
		
		# 3. Shortest Path Encoding (Section 3.3.1)
		# Computed separately and stored in dict for efficiency
		if args.use_spe:
			self.shortest_paths_dict = self.computeShortestPaths(adj_matrix)
		
		log('Positional encodings computed', level='SUCCESS')
		
		return {
			'degrees': degrees_tensor,
			'pagerank': pagerank_tensor,
			'adj_matrix': adj_matrix
		}
	
	def getSPE(self, central_node, sampled_nodes):
		"""
		Get shortest path distances from central_node to sampled_nodes
		
		Args:
			central_node: int, central node index
			sampled_nodes: tensor [k], sampled node indices
		
		Returns:
			distances: tensor [k+1, 1] (includes distance to self = 0)
		"""
		if not args.use_spe or self.shortest_paths_dict is None:
			# Fallback: uniform distances
			k = len(sampled_nodes)
			distances = t.ones(k + 1, 1) * 2.0
			distances[0, 0] = 0.0  # Distance to self
			return distances
		
		distances = []
		
		# Distance to self
		distances.append(0.0)
		
		# Distances to sampled nodes
		if central_node in self.shortest_paths_dict:
			paths = self.shortest_paths_dict[central_node]
			for sample_node in sampled_nodes:
				sample_node = sample_node.item() if t.is_tensor(sample_node) else sample_node
				dist = paths.get(sample_node, args.max_spe_distance)
				distances.append(float(dist))
		else:
			# Central node not in precomputed set, use average distance
			distances.extend([3.0] * len(sampled_nodes))
		
		return t.tensor(distances).unsqueeze(1).float()
	
	def computeAttentionSamples(self, embeddings, adj_matrix, k=20, alpha=0.5):
		"""
		Compute attention samples (Section 3.2, Equations 1-2) - VERSION OPTIMISÉE
		"""
		log(f'Computing attention samples (k={k})...', level='INFO')
		
		N = embeddings.shape[0]
		device = embeddings.device
		
		# ✅ OPTIMISATION : Traiter par plus gros batchs
		batch_size = 4096  # Au lieu de 2048
		attention_samples = []
		
		num_batches = (N + batch_size - 1) // batch_size
		
		# Convert adj to torch sparse
		adj_indices = t.from_numpy(np.vstack([adj_matrix.row, adj_matrix.col])).long()
		adj_values = t.from_numpy(adj_matrix.data).float()
		adj_shape = t.Size(adj_matrix.shape)
		adj_torch = t.sparse_coo_tensor(adj_indices, adj_values, adj_shape).to(device)
		
		# ✅ Pré-calculer A + I une seule fois
		identity_indices = t.stack([t.arange(N), t.arange(N)]).to(device)
		identity_values = t.ones(N).to(device)
		
		adj_with_self = t.sparse_coo_tensor(
			t.cat([adj_indices.to(device), identity_indices], dim=1),
			t.cat([adj_values.to(device), identity_values]),
			adj_shape
		).coalesce()
		
		for batch_idx in range(num_batches):
			start_idx = batch_idx * batch_size
			end_idx = min((batch_idx + 1) * batch_size, N)
			
			batch_embeds = embeddings[start_idx:end_idx]
			
			# Semantic similarity
			S_batch = t.mm(batch_embeds, embeddings.t())  # [batch, N]
			
			# Structure-aware update (Equation 2)
			if alpha > 0:
				# ✅ Utiliser sparse matmul directement
				batch_adj_dense = adj_with_self.to_dense()[start_idx:end_idx, :]
				neighbor_sim = t.mm(batch_adj_dense, S_batch.t()).t()
				S_batch = S_batch + alpha * neighbor_sim
			
			# Sample top-k
			_, top_k_indices = t.topk(S_batch, k, dim=1)
			attention_samples.append(top_k_indices.cpu())
			
			if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
				log(f'Attention sampling: {batch_idx + 1}/{num_batches} batches', 
					level='DEBUG', save=False, oneline=True)
		
		print()
		attention_samples = t.cat(attention_samples, dim=0).to(device)
		
		log(f'Attention samples computed: {attention_samples.shape}', level='SUCCESS')
		return attention_samples

	def LoadData(self):
		log('Loading data...', level='INFO')
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape
		
		log(f'Dataset: {args.data}', level='INFO')
		log(f'Users: {args.user}, Items: {args.item}', level='INFO')
		log(f'Train interactions: {trnMat.nnz}, Test: {tstMat.nnz}', level='INFO')
		log(f'Sparsity: {trnMat.nnz / (args.user * args.item) * 100:.4f}%', level='INFO')
		
		# Create adjacency matrix
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		
		# Create full adjacency for positional encodings
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		full_adj = sp.vstack([sp.hstack([a, trnMat]), sp.hstack([trnMat.transpose(), b])])
		full_adj = (full_adj != 0) * 1.0
		
		# Compute positional encodings
		self.pos_encodings = self.computePositionalEncodings(full_adj)
		
		# Create data loaders
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(
			trnData, 
			batch_size=args.batch, 
			shuffle=True, 
			num_workers=args.num_workers
		)
		
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(
			tstData, 
			batch_size=args.tstBat, 
			shuffle=False, 
			num_workers=args.num_workers
		)
		
		log('Data loading complete', level='SUCCESS')


class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])