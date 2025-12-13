import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	
	# Training parameters
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=2048, type=int, help='batch size (reduced for Colab)')
	parser.add_argument('--epoch', default=50, type=int, help='number of epochs (reduced for faster testing)')
	parser.add_argument('--decay', default=0, type=float, help='weight decay rate')
	parser.add_argument('--tstEpoch', default=5, type=int, help='test every N epochs')
	parser.add_argument('--tstBat', default=128, type=int, help='test batch size')
	
	# Model architecture parameters
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')
	parser.add_argument('--block_num', default=2, type=int, help='number of TransGNN blocks')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers (legacy)')
	parser.add_argument('--num_head', default=4, type=int, help='number of attention heads in transformer')
	parser.add_argument('--att_head', default=2, type=int, help='number of attention heads (legacy)')
	parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate for transformer layers')
	
	# Attention sampling parameters
	parser.add_argument('--k_samples', default=20, type=int, help='number of attention samples (k)')
	parser.add_argument('--alpha', default=0.5, type=float, help='alpha parameter for attention sampling')
	
	# Positional encoding parameters
	parser.add_argument('--use_spe', default=True, type=bool, help='use shortest path encoding')
	parser.add_argument('--use_de', default=True, type=bool, help='use degree encoding')
	parser.add_argument('--use_pre', default=True, type=bool, help='use pagerank encoding')
	
	# Evaluation parameters
	parser.add_argument('--topk', default=20, type=int, help='K of top K for evaluation')
	
	# Legacy parameters (kept for compatibility)
	parser.add_argument('--leaky', default=0.5, type=float, help='slope of leaky relu')
	parser.add_argument('--hyperNum', default=128, type=int, help='number of hyperedges')
	parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')
	parser.add_argument('--temp', default=0.2, type=float, help='temperature for contrastive loss')
	parser.add_argument('--mult', default=1, type=float, help='multiplication factor')
	parser.add_argument('--edgeSampRate', default=0.1, type=float, help='ratio of sampled edges')
	
	# Dataset and paths
	parser.add_argument('--data', default='yelp', type=str, 
	                    help='dataset name (yelp/ml10m/tmall/gowalla/amazon-book)')
	parser.add_argument('--save_path', default='run', type=str,
	                    help='experiment name for results file')
	parser.add_argument('--load_model', default=None, type=str,
	                    help='model name to load (not used in clean version)')
	
	# System parameters
	parser.add_argument('--gpu', default='0', type=str, help='GPU device ID to use')
	parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')
	
	return parser.parse_args()

args = ParseArgs()