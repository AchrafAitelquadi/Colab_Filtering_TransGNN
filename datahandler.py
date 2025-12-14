import pickle
import numpy as np
from scipy.sparse import coo_matrix
from params import args
import scipy.sparse as sp
from utils.timeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader

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
		self.shortest_paths_dict = None

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
		SIMPLIFIED SPE - Much faster, approximate version
		"""
		log('Computing Shortest Path Encoding (SPE)...', level='INFO')
		
		N = adj_matrix.shape[0]
		
		# Reduce sample size drastically for speed
		sample_size = min(args.spe_sample_size, 1000)
		sample_sources = np.random.choice(N, sample_size, replace=False)
		
		log(f'   SPE sample size: {len(sample_sources)} nodes (fast mode)', level='INFO')
		
		# Convert to CSR
		adj_csr = adj_matrix.tocsr()
		
		shortest_paths_dict = {}
		max_hops = 3
		
		for idx, source in enumerate(sample_sources):
			distances = {source: 0}
			
			# Simple BFS with early stopping
			current_level = {source}
			visited = {source}
			
			for hop in range(1, max_hops + 1):
				next_level = set()
				
				for node in current_level:
					neighbors = adj_csr.getrow(node).nonzero()[1]
					
					for neighbor in neighbors[:50]:
						if neighbor not in visited:
							visited.add(neighbor)
							next_level.add(neighbor)
							distances[neighbor] = hop
							
							if len(next_level) >= 100:
								break
					
					if len(next_level) >= 100:
						break
				
				if len(next_level) == 0:
					break
				
				current_level = next_level
			
			shortest_paths_dict[source] = distances
			
			# Progress every 100 nodes
			if (idx + 1) % 100 == 0 or (idx + 1) == len(sample_sources):
				progress = (idx + 1) / len(sample_sources) * 100
				print(f'\r   SPE Progress: [{idx + 1}/{len(sample_sources)}] ({progress:.1f}%)', 
					  end='', flush=True)
		
		print()  # New line
		log(f'✓ SPE computed for {len(shortest_paths_dict)} nodes', level='SUCCESS')
		
		return shortest_paths_dict

	def computePositionalEncodings(self, adj_matrix):
		"""
		Compute positional encodings - OPTIMIZED VERSION
		"""
		log('Computing positional encodings...', level='INFO')
		
		N = adj_matrix.shape[0]
		device = t.device('cuda' if t.cuda.is_available() else 'cpu')
		
		# 1. Degree Encoding
		degrees = np.array(adj_matrix.sum(axis=1)).flatten()
		degrees_tensor = t.from_numpy(degrees).float().unsqueeze(1).to(device)
		log('   ✓ Degree encoding computed', level='INFO')
		
		# 2. PageRank Encoding (simplified)
		total_degree = degrees.sum() + 1e-8
		pagerank_values = degrees / total_degree
		pagerank_tensor = t.from_numpy(pagerank_values).float().unsqueeze(1).to(device)
		log('   ✓ PageRank encoding computed', level='INFO')
		
		# 3. Shortest Path Encoding (simplified and cached)
		if args.use_spe:
			self.shortest_paths_dict = self.computeShortestPaths(adj_matrix)
		
		log('✓ All positional encodings ready', level='SUCCESS')
		
		return {
			'degrees': degrees_tensor,
			'pagerank': pagerank_tensor,
			'adj_matrix': adj_matrix
		}
	
	def getSPE(self, central_node, sampled_nodes):
		"""
		Get shortest path distances - OPTIMIZED with fallback
		"""
		if not args.use_spe or self.shortest_paths_dict is None:
			k = len(sampled_nodes)
			distances = t.ones(k + 1, 1) * 2.0
			distances[0, 0] = 0.0
			return distances
		
		distances = [0.0]  # Distance to self
		
		# Fast lookup
		if central_node in self.shortest_paths_dict:
			paths = self.shortest_paths_dict[central_node]
			for sample_node in sampled_nodes:
				sample_node = sample_node.item() if t.is_tensor(sample_node) else sample_node
				dist = paths.get(sample_node, 3.0)
				distances.append(float(dist))
		else:
			# Fallback
			distances.extend([2.5] * len(sampled_nodes))
		
		return t.tensor(distances).unsqueeze(1).float()
	
	def computeAttentionSamples(self, embeddings, adj_matrix, k=20, alpha=0.5):
		"""
		Compute attention samples - HIGHLY OPTIMIZED VERSION
		"""
		log(f'Computing attention samples (k={k})...', level='INFO')
		
		N = embeddings.shape[0]
		device = embeddings.device
		
		batch_size = 8192
		attention_samples = []
		
		num_batches = (N + batch_size - 1) // batch_size
		
		# Pre-compute adjacency on GPU
		adj_indices = t.from_numpy(np.vstack([adj_matrix.row, adj_matrix.col])).long().to(device)
		adj_values = t.from_numpy(adj_matrix.data).float().to(device)
		adj_shape = t.Size(adj_matrix.shape)
		adj_torch = t.sparse_coo_tensor(adj_indices, adj_values, adj_shape).coalesce()
		
		# Pre-compute identity
		identity_indices = t.stack([t.arange(N, device=device), t.arange(N, device=device)])
		identity_values = t.ones(N, device=device)
		
		adj_with_self = t.sparse_coo_tensor(
			t.cat([adj_indices, identity_indices], dim=1),
			t.cat([adj_values, identity_values]),
			adj_shape
		).coalesce()
		
		# Process in large batches
		for batch_idx in range(num_batches):
			start_idx = batch_idx * batch_size
			end_idx = min((batch_idx + 1) * batch_size, N)
			
			batch_embeds = embeddings[start_idx:end_idx]
			
			# Semantic similarity
			S_batch = t.mm(batch_embeds, embeddings.t())
			
			# Structure-aware update
			if alpha > 0:
				neighbor_sim = t.sparse.mm(adj_with_self, S_batch.t()).t()
				S_batch = S_batch + alpha * neighbor_sim
			
			# Sample top-k
			_, top_k_indices = t.topk(S_batch, k, dim=1)
			attention_samples.append(top_k_indices.cpu())
			
			# Progress every 2 batches
			if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == num_batches:
				progress = (batch_idx + 1) / num_batches * 100
				print(f'\r   Sampling: [{batch_idx + 1}/{num_batches}] ({progress:.1f}%)', 
					  end='', flush=True)
		
		print()  # New line
		attention_samples = t.cat(attention_samples, dim=0).to(device)
		
		log(f'✓ Attention samples: {attention_samples.shape}', level='SUCCESS')
		return attention_samples

	def LoadData(self):
		log('Loading data...', level='INFO')
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape
		
		log(f'Dataset: {args.data}', level='INFO')
		log(f'   Users: {args.user}, Items: {args.item}', level='INFO')
		log(f'   Train interactions: {trnMat.nnz}, Test: {tstMat.nnz}', level='INFO')
		log(f'   Sparsity: {trnMat.nnz / (args.user * args.item) * 100:.4f}%', level='INFO')
		
		# Create adjacency matrix
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		log('✓ Adjacency matrix created', level='SUCCESS')
		
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
			num_workers=args.num_workers,
			pin_memory=True
		)
		
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(
			tstData, 
			batch_size=args.tstBat, 
			shuffle=False, 
			num_workers=args.num_workers,
			pin_memory=True
		)
		
		log('✓ Data loading complete', level='SUCCESS')
		log('='*60, level='INFO')


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