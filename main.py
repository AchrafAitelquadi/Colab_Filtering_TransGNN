import torch as t
from utils.timeLogger import log
from params import args
from model import TransGNN
from datahandler import DataHandler
import numpy as np
import os
import random
import time


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


class Coach:
    def __init__(self, handler):
        self.handler = handler
        
        log('='*60, level='INFO')
        log('TransGNN - Transformer + GNN for Recommendation', level='SUCCESS')
        log('='*60, level='INFO')
        log(f'Dataset: {args.data.upper()}', level='INFO')
        log(f'Users: {args.user}, Items: {args.item}', level='INFO')
        log(f'Interactions: {handler.trnLoader.dataset.__len__()}', level='INFO')
        log(f'Sparsity: {handler.trnLoader.dataset.__len__() / (args.user * args.item) * 100:.4f}%', level='INFO')
        log('='*60, level='INFO')
        log(f'Model Config:', level='INFO')
        log(f'  - Embedding dim: {args.latdim}', level='INFO')
        log(f'  - Architecture: Trans(3) + GNN(2) [FIXED]', level='INFO')
        log(f'  - Attention heads: {args.num_head}', level='INFO')
        log(f'  - Attention samples (k): {args.k_samples}', level='INFO')
        log(f'  - Dropout: {args.dropout}', level='INFO')
        log(f'  - Learning rate: {args.lr}', level='INFO')
        log(f'  - Sample update: {args.update_strategy}', level='INFO')
        log('='*60, level='INFO')
        
        self.attention_samples = None
        self.best_recall = 0
        self.best_ndcg = 0
        self.best_epoch = 0
        
        self.results = {
            'train_losses': [],
            'test_recalls': [],
            'test_ndcgs': [],
            'epochs': []
        }

    def run(self):
        """Main training loop"""
        self.prepareModel()
        
        # Initialize attention samples UNE SEULE FOIS
        log('Initializing attention samples...', level='INFO')
        with t.no_grad():
            init_embeds = t.cat([self.model.user_embedding, self.model.item_embedding], dim=0)
            
            import scipy.sparse as sp
            adj_indices = self.handler.torchBiAdj._indices().cpu().numpy()
            adj_values = self.handler.torchBiAdj._values().cpu().numpy()
            adj_sparse = sp.coo_matrix(
                (adj_values, (adj_indices[0], adj_indices[1])),
                shape=self.handler.torchBiAdj.shape
            )
            
            self.attention_samples = self.handler.computeAttentionSamples(
                init_embeds, adj_sparse, k=args.k_samples, alpha=args.alpha
            )
            if t.cuda.is_available():
                self.attention_samples = self.attention_samples.cuda()
        
        log('Starting training...', level='SUCCESS')
        log('='*60, level='INFO')
        
        start_time = time.time()
        no_improve_count = 0
        
        for ep in range(args.epoch):
            epoch_start = time.time()
            
            # Training
            train_loss = self.trainEpoch()
            train_time = time.time() - epoch_start
            
            self.results['train_losses'].append(train_loss)
            
            log(f'Epoch {ep:3d}/{args.epoch} | Loss: {train_loss:.4f} | Time: {train_time:.1f}s', 
                level='INFO')
            
            # Testing
            if ep % args.tstEpoch == 0:
                test_start = time.time()
                recall, ndcg = self.testEpoch()
                test_time = time.time() - test_start
                
                self.results['test_recalls'].append(recall)
                self.results['test_ndcgs'].append(ndcg)
                self.results['epochs'].append(ep)
                
                # Check improvement
                if recall > self.best_recall:
                    self.best_recall = recall
                    self.best_ndcg = ndcg
                    self.best_epoch = ep
                    no_improve_count = 0
                    
                    log(f'✨ NEW BEST! Recall@{args.topk}: {recall:.4f}, NDCG@{args.topk}: {ndcg:.4f}', 
                        level='SUCCESS')
                    
                    self.saveModel('best')
                else:
                    no_improve_count += 1
                
                log(f'Test | Recall@{args.topk}: {recall:.4f}, NDCG@{args.topk}: {ndcg:.4f} | Time: {test_time:.1f}s', 
                    level='INFO')
                log(f'Best so far: Recall@{args.topk}: {self.best_recall:.4f} (Epoch {self.best_epoch})', 
                    level='INFO')
                
                # Early stopping
                if no_improve_count >= 10:
                    log(f'Early stopping: no improvement for {no_improve_count} test epochs', level='WARN')
                    break
            
            # ✅ PLUS D'UPDATE PÉRIODIQUE ICI - C'est fait dans le forward() !
        
        # Final test
        log('='*60, level='INFO')
        log('Running final evaluation...', level='INFO')
        final_recall, final_ndcg = self.testEpoch()
        
        total_time = time.time() - start_time
        
        log('='*60, level='SUCCESS')
        log('Training Complete!', level='SUCCESS')
        log('='*60, level='SUCCESS')
        log(f'Total time: {total_time/60:.2f} minutes', level='INFO')
        log(f'Best Results (Epoch {self.best_epoch}):', level='SUCCESS')
        log(f'  - Recall@{args.topk}: {self.best_recall:.4f}', level='SUCCESS')
        log(f'  - NDCG@{args.topk}: {self.best_ndcg:.4f}', level='SUCCESS')
        log(f'Final Results:', level='INFO')
        log(f'  - Recall@{args.topk}: {final_recall:.4f}', level='INFO')
        log(f'  - NDCG@{args.topk}: {final_ndcg:.4f}', level='INFO')
        log('='*60, level='SUCCESS')
        
        self.saveResults()
        
        return self.results
    
    def prepareModel(self):
        """Initialize model and optimizer"""
        self.model = TransGNN()
        if t.cuda.is_available():
            self.model = self.model.cuda()
        
        self.opt = t.optim.Adam(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.decay
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        log(f'Model initialized:', level='SUCCESS')
        log(f'  - Total parameters: {total_params:,}', level='INFO')
        log(f'  - Trainable parameters: {trainable_params:,}', level='INFO')
    
    def trainEpoch(self):
        """Train one epoch"""
        self.model.train()
        self.handler.trnLoader.dataset.negSampling()
        
        epoch_loss = 0
        steps = len(self.handler.trnLoader)
        
        log(f'Starting training epoch with {steps} batches...', level='DEBUG')
        
        for i, (ancs, poss, negs) in enumerate(self.handler.trnLoader):
            if i == 0:
                log(f'Processing first batch: ancs={ancs.shape}, poss={poss.shape}, negs={negs.shape}', level='DEBUG')
            
            # Move to device
            if t.cuda.is_available():
                ancs = ancs.long().cuda()
                poss = poss.long().cuda()
                negs = negs.long().cuda()
            else:
                ancs = ancs.long()
                poss = poss.long()
                negs = negs.long()
            
            if i == 0:
                log(f'Data moved to device', level='DEBUG')
            
            # Get adjacency matrix
            adj = self.handler.torchBiAdj
            if t.cuda.is_available():
                adj = adj.cuda()
            
            if i == 0:
                log(f'Starting forward pass...', level='DEBUG')
            
            # Forward pass
            try:
                loss = self.model.calcLosses(
                    ancs, poss, negs, adj,
                    attention_samples=self.attention_samples,
                    handler=self.handler
                )
            except Exception as e:
                log(f'Error in forward pass at batch {i}: {e}', level='ERROR')
                raise
            
            if i == 0:
                log(f'Forward pass complete, loss={loss.item():.4f}', level='DEBUG')
            
            epoch_loss += loss.item()
            
            # Backward pass
            self.opt.zero_grad()
            
            if i == 0:
                log(f'Starting backward pass...', level='DEBUG')
            
            loss.backward()
            
            if i == 0:
                log(f'Backward pass complete', level='DEBUG')
            
            self.opt.step()
            
            if i == 0:
                log(f'Optimizer step complete', level='DEBUG')
            
            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == steps:
                print(f'\r  Train Progress: [{i+1}/{steps}] Loss: {loss.item():.4f}', 
                      end='', flush=True)
        
        print()  # New line
        return epoch_loss / steps

    def testEpoch(self):
        """Evaluate on test set"""
        self.model.eval()
        
        epoch_recall = 0
        epoch_ndcg = 0
        num_users = 0
        steps = len(self.handler.tstLoader)
        
        adj = self.handler.torchBiAdj
        if t.cuda.is_available():
            adj = adj.cuda()
        
        with t.no_grad():
            for i, (usr, trnMask) in enumerate(self.handler.tstLoader):
                # Move to device
                if t.cuda.is_available():
                    usr = usr.long().cuda()
                    trnMask = trnMask.cuda()
                else:
                    usr = usr.long()
                
                # Get embeddings
                usrEmbeds, itmEmbeds = self.model.predict(
                    adj,
                    attention_samples=self.attention_samples,
                    handler=self.handler
                )
                
                # Compute scores
                allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0))
                
                # Mask training items
                allPreds = allPreds * (1 - trnMask) - trnMask * 1e8
                
                # Get top-K
                _, topLocs = t.topk(allPreds, args.topk)
                
                # Calculate metrics
                recall, ndcg = self.calcRes(
                    topLocs.cpu().numpy(),
                    self.handler.tstLoader.dataset.tstLocs,
                    usr.cpu().numpy()
                )
                
                epoch_recall += recall
                epoch_ndcg += ndcg
                num_users += len(usr)
                
                # Progress
                if (i + 1) % 20 == 0 or (i + 1) == steps:
                    print(f'\r  Test Progress: [{i+1}/{steps}]', end='', flush=True)
        
        print()  # New line
        
        avg_recall = epoch_recall / num_users
        avg_ndcg = epoch_ndcg / num_users
        
        return avg_recall, avg_ndcg

    def calcRes(self, topLocs, tstLocs, batIds):
        """Calculate Recall and NDCG"""
        all_recall = 0
        all_ndcg = 0
        
        for i in range(len(batIds)):
            tem_top_locs = list(topLocs[i])
            tem_tst_locs = tstLocs[batIds[i]]
            
            if tem_tst_locs is None:
                continue
            
            tst_num = len(tem_tst_locs)
            max_dcg = np.sum([np.reciprocal(np.log2(loc + 2)) 
                              for loc in range(min(tst_num, args.topk))])
            
            recall = 0
            dcg = 0
            
            for val in tem_tst_locs:
                if val in tem_top_locs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(tem_top_locs.index(val) + 2))
            
            recall = recall / tst_num if tst_num > 0 else 0
            ndcg = dcg / max_dcg if max_dcg > 0 else 0
            
            all_recall += recall
            all_ndcg += ndcg
        
        return all_recall, all_ndcg
    
    def saveModel(self, name):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        save_path = f'checkpoints/{args.data}_{args.save_path}_{name}.pth'
        
        t.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'best_recall': self.best_recall,
            'best_ndcg': self.best_ndcg,
            'attention_samples': self.attention_samples,
        }, save_path)
        
        log(f'Model saved: {save_path}', level='DEBUG')
    
    def saveResults(self):
        """Save training results"""
        import json
        
        os.makedirs('results', exist_ok=True)
        
        results_dict = {
            'dataset': args.data,
            'config': {
                'latdim': args.latdim,
                'num_head': args.num_head,
                'k_samples': args.k_samples,
                'dropout': args.dropout,
                'lr': args.lr,
                'update_strategy': args.update_strategy,
            },
            'best_results': {
                'epoch': self.best_epoch,
                'recall': self.best_recall,
                'ndcg': self.best_ndcg,
            },
            'training_history': {
                'train_losses': self.results['train_losses'],
                'test_recalls': self.results['test_recalls'],
                'test_ndcgs': self.results['test_ndcgs'],
                'test_epochs': self.results['epochs'],
            }
        }
        
        save_path = f'results/{args.data}_{args.save_path}_results.json'
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        log(f'Results saved: {save_path}', level='INFO')


if __name__ == '__main__':
    # Set seed
    set_seed(args.seed)
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Print configuration
    log('='*60, level='INFO')
    log('Configuration:', level='INFO')
    log(f'  Dataset: {args.data}', level='INFO')
    log(f'  GPU: {args.gpu}', level='INFO')
    log(f'  Seed: {args.seed}', level='INFO')
    log(f'  Batch size: {args.batch}', level='INFO')
    log(f'  Epochs: {args.epoch}', level='INFO')
    log('='*60, level='INFO')
    
    # Load data
    handler = DataHandler()
    handler.LoadData()
    
    # Train
    coach = Coach(handler)
    results = coach.run()
    
    log('✅ Experiment complete!', level='SUCCESS')