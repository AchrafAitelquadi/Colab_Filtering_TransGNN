import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='TransGNN - Official Implementation')
	
	# ==================== TRAINING PARAMETERS ====================
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=4096, type=int, 
	                    help='batch size (paper uses 4096)')
	parser.add_argument('--epoch', default=50, type=int, 
	                    help='max epochs (paper uses early stopping)')
	parser.add_argument('--decay', default=0, type=float, help='weight decay')
	parser.add_argument('--tstEpoch', default=3, type=int, 
	                    help='test every N epochs')
	parser.add_argument('--tstBat', default=64, type=int, help='test batch size')
	
	# ==================== MODEL ARCHITECTURE ====================
	# Section 4.1.4: "We use three Transformer layers with two GNN layers sandwiched between them"
	# Architecture: Trans -> GNN -> Trans -> GNN -> Trans (3 Trans, 2 GNN)
	parser.add_argument('--latdim', default=32, type=int, 
	                    help='embedding dimension - paper uses 32 or 64')
	parser.add_argument('--trans_layers', default=3, type=int, 
	                    help='number of Transformer layers (paper: 3)')
	parser.add_argument('--gnn_layers', default=2, type=int, 
	                    help='number of GNN layers (paper: 2)')
	parser.add_argument('--num_head', default=4, type=int, 
	                    help='number of attention heads in Transformer')
	parser.add_argument('--dropout', default=0.1, type=float, 
	                    help='dropout rate')
	
	# ==================== ATTENTION SAMPLING (Section 3.2) ====================
	parser.add_argument('--k_samples', default=20, type=int, 
	                    help='number of attention samples (k) - paper uses 20-30')
	parser.add_argument('--alpha', default=0.5, type=float, 
	                    help='balance factor for structure-aware similarity (Eq. 2)')
	
	# ==================== POSITIONAL ENCODING (Section 3.3) ====================
	parser.add_argument('--use_spe', default=True, type=bool, 
	                    help='use Shortest Path Encoding (SPE)')
	parser.add_argument('--use_de', default=True, type=bool, 
	                    help='use Degree Encoding (DE)')
	parser.add_argument('--use_pre', default=True, type=bool, 
	                    help='use PageRank Encoding (PRE)')
	parser.add_argument('--max_spe_distance', default=10, type=int, 
	                    help='maximum shortest path distance to consider')
	
	# ==================== SAMPLE UPDATE (Section 3.4.3) ====================
	parser.add_argument('--update_strategy', default='message_passing', 
	                    choices=['message_passing', 'random_walk', 'none'],
	                    help='strategy for updating attention samples')
	parser.add_argument('--update_frequency', default=1, type=int, 
	                    help='update samples every N blocks')
	parser.add_argument('--random_walk_length', default=5, type=int, 
	                    help='length of random walk for RW update')
	
	# ==================== EVALUATION ====================
	parser.add_argument('--topk', default=20, type=int, 
	                    help='K for Recall@K and NDCG@K')
	
	# ==================== DATASET ====================
	parser.add_argument('--data', default='yelp', type=str, 
	                    choices=['yelp', 'gowalla', 'tmall', 'amazon-book', 'ml10m'],
	                    help='dataset name')
	parser.add_argument('--save_path', default='exp1', type=str,
	                    help='experiment identifier')
	
	# ==================== SYSTEM ====================
	parser.add_argument('--gpu', default='0', type=str, help='GPU device ID')
	parser.add_argument('--seed', default=42, type=int, help='random seed')
	parser.add_argument('--num_workers', default=2, type=int, help='dataloader workers')
	
	# ==================== PREPROCESSING ====================
	parser.add_argument('--precompute_spe', default=True, type=bool,
	                    help='precompute shortest paths (slow but accurate)')
	parser.add_argument('--spe_sample_size', default=5000, type=int,
	                    help='number of nodes to sample for SPE computation')
	

	parser.add_argument('--update_every_block', default=True, type=bool,
                    help='Update attention samples after each Transformer block')
	return parser.parse_args()

args = ParseArgs()
