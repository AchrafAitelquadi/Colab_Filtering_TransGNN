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
		
		# Positional encodings will be computed after loading data
		self.pos_encodings = None

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
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		# mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor - use new API
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		
		# Use the new API instead of deprecated SparseTensor
		sparse_tensor = t.sparse_coo_tensor(idxs, vals, shape)
		return sparse_tensor.cuda() if t.cuda.is_available() else sparse_tensor

	def makeSample(self):
		user_sample_idx = t.tensor([[args.user + i for i in range(args.item)] * args.user])
		item_sample_idx = t.tensor([[i for i in range(args.user)] * args.item])
		return user_sample_idx, item_sample_idx

	def makeMask(self):
		"""Memory-efficient mask - handled during evaluation"""
		return None

	def computePositionalEncodings(self, adj_matrix):
		"""Fast positional encodings"""
		log('Computing positional encodings...', level='INFO')
		
		N = adj_matrix.shape[0]
		device = t.device('cuda' if t.cuda.is_available() else 'cpu')
		
		# Degree-based encoding
		degrees = np.array(adj_matrix.sum(axis=1)).flatten()
		degrees_tensor = t.from_numpy(degrees).float().unsqueeze(1).to(device)
		
		# Fast PageRank approximation
		pagerank_values = degrees / (degrees.sum() + 1e-8)
		pagerank_tensor = t.from_numpy(pagerank_values).float().unsqueeze(1).to(device)
		
		return {
			'degrees': degrees_tensor,
			'pagerank': pagerank_tensor,
			'adj_matrix': adj_matrix
		}
	
	def computeAttentionSamples(self, embeddings, adj_matrix, k=20, alpha=0.5):
		"""
		Compute attention samples based on semantic similarity and graph structure
		Memory-efficient version
		
		Args:
			embeddings: node embeddings [N, d]
			adj_matrix: adjacency matrix (scipy sparse)
			k: number of samples to keep
			alpha: balance factor for structure-aware update
			
		Returns:
			attention_samples: [N, k] indices of sampled nodes
		"""
		log(f'Computing attention samples with k={k} (memory-efficient mode)...', level='INFO')
		
		N = embeddings.shape[0]
		device = embeddings.device
		
		# Method 1: Pure semantic similarity (most memory efficient)
		# Process in batches to compute similarity
		batch_size = 4096
		attention_samples = []
		
		num_batches = (N + batch_size - 1) // batch_size
		
		for batch_idx in range(num_batches):
			start_idx = batch_idx * batch_size
			end_idx = min((batch_idx + 1) * batch_size, N)
			
			# Get batch embeddings
			batch_embeds = embeddings[start_idx:end_idx]
			
			# Compute similarity for this batch: S = X_batch @ X^T
			S_batch = t.mm(batch_embeds, embeddings.t())  # [batch_size, N]
			
			# Sample top-k for this batch based on semantic similarity
			_, top_k_indices = t.topk(S_batch, k, dim=1)
			attention_samples.append(top_k_indices.cpu())
			
			if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
				log(f'Attention sampling: {batch_idx + 1}/{num_batches} batches', 
					level='DEBUG', save=False, oneline=True)
		
		print()  # New line after progress
		
		# Concatenate all batches
		attention_samples = t.cat(attention_samples, dim=0).to(device)
		
		log(f'Attention samples computed: shape {attention_samples.shape}', level='SUCCESS')
		
		return attention_samples

	def LoadData(self):
		log('Loading data...', level='INFO')
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape
		
		log(f'Data: {args.user} users, {args.item} items, {trnMat.nnz} train, {tstMat.nnz} test', level='SUCCESS')
		
		# Create adjacency matrix
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		self.mask = self.makeMask()
		
		# Compute positional encodings
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		full_adj = sp.vstack([sp.hstack([a, trnMat]), sp.hstack([trnMat.transpose(), b])])
		full_adj = (full_adj != 0) * 1.0
		
		self.pos_encodings = self.computePositionalEncodings(full_adj)
		
		# Create data loaders
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)
		
		log('Data loading complete', level='SUCCESS')

class TrnMaskedData(data.Dataset):
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