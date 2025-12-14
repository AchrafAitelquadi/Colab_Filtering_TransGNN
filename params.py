import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='TransGNN - Optimized Implementation')
	
	# ==================== TRAINING PARAMETERS ====================
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=4096, type=int, help='batch size')
	parser.add_argument('--epoch', default=40, type=int, help='max epochs')
	parser.add_argument('--decay', default=0, type=float, help='weight decay')
	parser.add_argument('--tstEpoch', default=3, type=int, help='test every N epochs')
	parser.add_argument('--tstBat', default=256, type=int, help='test batch size')
	
	# ==================== MODEL ARCHITECTURE ====================
	parser.add_argument('--latdim', default=32, type=int, help='embedding dimension')
	parser.add_argument('--trans_layers', default=3, type=int, help='number of Transformer layers')
	parser.add_argument('--gnn_layers', default=2, type=int, help='number of GNN layers')
	parser.add_argument('--num_head', default=4, type=int, help='number of attention heads')
	parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
	
	# ==================== ATTENTION SAMPLING ====================
	parser.add_argument('--k_samples', default=20, type=int, help='number of attention samples')
	parser.add_argument('--alpha', default=0.5, type=float, help='balance factor for structure-aware similarity')
	
	# ==================== POSITIONAL ENCODING (OPTIMIZED) ====================
	parser.add_argument('--use_spe', default=True, type=bool, help='use Shortest Path Encoding')
	parser.add_argument('--use_de', default=True, type=bool, help='use Degree Encoding')
	parser.add_argument('--use_pre', default=True, type=bool, help='use PageRank Encoding')
	parser.add_argument('--max_spe_distance', default=3, type=int, 
	                    help='max shortest path distance (reduced for speed)')
	
	# ==================== SAMPLE UPDATE ====================
	parser.add_argument('--update_strategy', default='none', 
	                    choices=['message_passing', 'random_walk', 'none'],
	                    help='OPTIMIZATION: Set to none for faster training')
	parser.add_argument('--update_frequency', default=1, type=int, help='update samples every N blocks')
	parser.add_argument('--random_walk_length', default=5, type=int, help='length of random walk')
	
	# ==================== EVALUATION ====================
	parser.add_argument('--topk', default=20, type=int, help='K for Recall@K and NDCG@K')
	
	# ==================== DATASET ====================
	parser.add_argument('--data', default='yelp', type=str, 
	                    choices=['yelp', 'gowalla', 'tmall', 'amazon-book', 'ml10m'],
	                    help='dataset name')
	parser.add_argument('--save_path', default='exp_optimized', type=str, help='experiment identifier')
	
	# ==================== SYSTEM ====================
	parser.add_argument('--gpu', default='0', type=str, help='GPU device ID')
	parser.add_argument('--seed', default=42, type=int, help='random seed')
	parser.add_argument('--num_workers', default=2, type=int, help='dataloader workers (2 for Colab)')
	
	# ==================== OPTIMIZATION FLAGS ====================
	parser.add_argument('--precompute_spe', default=True, type=bool,
	                    help='precompute shortest paths')
	parser.add_argument('--spe_sample_size', default=1000, type=int,
	                    help='CRITICAL: Reduced from 5000 to 1000 for speed')
	
	parser.add_argument('--update_every_block', default=False, type=bool,
	                    help='OPTIMIZATION: Disabled for faster training')
	
	# NEW: Performance optimization flags
	parser.add_argument('--use_mixed_precision', default=True, type=bool,
	                    help='Use automatic mixed precision for faster training')
	parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
	                    help='Gradient accumulation for larger effective batch size')
	parser.add_argument('--compile_model', default=False, type=bool,
	                    help='Use torch.compile for faster execution (PyTorch 2.0+)')
	
	return parser.parse_args()

args = ParseArgs()