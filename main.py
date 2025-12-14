import torch as t
from utils.timeLogger import log
from params import args
from model import TransGNN
from datahandler import DataHandler
from results_manager import ResultsManager
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
        
        # Initialize ResultsManager
        self.results_manager = ResultsManager()
        
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
        
        self.lr_history = []  # Track learning rate changes

    def run(self):
        """Main training loop"""
        self.prepareModel()
        
        # Initialize attention samples ONCE
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
        
        global_start_time = time.time()
        no_improve_count = 0
        
        for ep in range(args.epoch):
            epoch_start = time.time()
            
            # Get current learning rate
            current_lr = self.opt.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            
            # Training
            train_loss = self.trainEpoch(ep)
            train_time = time.time() - epoch_start
            
            # Log training results to CSV
            self.results_manager.log_epoch_results(
                epoch=ep + 1,
                phase='Train',
                results={'Loss': train_loss},
                lr=current_lr
            )
            
            log(f'Epoch [{ep+1:3d}/{args.epoch}] | Loss: {train_loss:.4f} | Time: {train_time:.1f}s | LR: {current_lr:.6f}', 
                level='INFO')
            
            # Testing
            if (ep + 1) % args.tstEpoch == 0 or (ep + 1) == args.epoch:
                test_start = time.time()
                recall, ndcg = self.testEpoch()
                test_time = time.time() - test_start
                
                # Log test results to CSV
                self.results_manager.log_epoch_results(
                    epoch=ep + 1,
                    phase='Test',
                    results={'Recall': recall, 'NDCG': ndcg},
                    lr=current_lr
                )
                
                # Check improvement
                improved = False
                if recall > self.best_recall:
                    self.best_recall = recall
                    self.best_ndcg = ndcg
                    self.best_epoch = ep
                    no_improve_count = 0
                    improved = True
                    
                    log(f'✨ NEW BEST! Recall@{args.topk}: {recall:.4f}, NDCG@{args.topk}: {ndcg:.4f}', 
                        level='SUCCESS')
                    
                    self.saveModel('best')
                else:
                    no_improve_count += 1
                
                # Display test results
                status_symbol = '✨' if improved else '  '
                log(f'{status_symbol} Epoch [{ep+1:3d}/{args.epoch}] | TEST | Recall@{args.topk}: {recall:.4f} | NDCG@{args.topk}: {ndcg:.4f} | Time: {test_time:.1f}s', 
                    level='SUCCESS' if improved else 'INFO')
                log(f'   Best so far: Recall@{args.topk}: {self.best_recall:.4f}, NDCG@{args.topk}: {self.best_ndcg:.4f} (Epoch {self.best_epoch+1})', 
                    level='INFO')
                
                # Early stopping
                if no_improve_count >= 10:
                    log(f'Early stopping: no improvement for {no_improve_count} test epochs', level='WARN')
                    break
            
            log('-'*60, level='INFO')
        
        # Calculate total training time
        total_training_time = time.time() - global_start_time
        
        # Final summary
        log('='*60, level='SUCCESS')
        log('Training Complete!', level='SUCCESS')
        log('='*60, level='SUCCESS')
        log(f'Total training time: {total_training_time/60:.2f} minutes ({total_training_time:.1f}s)', level='INFO')
        log(f'Best Results (Epoch {self.best_epoch+1}):', level='SUCCESS')
        log(f'  - Recall@{args.topk}: {self.best_recall:.4f}', level='SUCCESS')
        log(f'  - NDCG@{args.topk}: {self.best_ndcg:.4f}', level='SUCCESS')
        log('='*60, level='SUCCESS')
        
        # Generate all visualizations and reports
        best_results = {
            'best_recall': self.best_recall,
            'best_ndcg': self.best_ndcg,
            'best_epoch': self.best_epoch + 1
        }
        
        output_files = self.results_manager.finalize(
            best_results=best_results,
            training_time=total_training_time
        )
        
        # Plot learning rate schedule
        if self.lr_history:
            lr_plot = self.results_manager.plot_learning_rate_schedule(self.lr_history)
            if lr_plot:
                log(f'Learning rate schedule saved: {lr_plot}', level='INFO')
        
        return {
            'best_recall': self.best_recall,
            'best_ndcg': self.best_ndcg,
            'best_epoch': self.best_epoch,
            'total_time': total_training_time,
            'output_files': output_files
        }
    
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
    
    def trainEpoch(self, epoch_num):
        """Train one epoch"""
        self.model.train()
        self.handler.trnLoader.dataset.negSampling()
        
        epoch_loss = 0
        batch_losses = []
        steps = len(self.handler.trnLoader)
        
        for i, (ancs, poss, negs) in enumerate(self.handler.trnLoader):
            # Move to device
            if t.cuda.is_available():
                ancs = ancs.long().cuda()
                poss = poss.long().cuda()
                negs = negs.long().cuda()
            else:
                ancs = ancs.long()
                poss = poss.long()
                negs = negs.long()
            
            # Get adjacency matrix
            adj = self.handler.torchBiAdj
            if t.cuda.is_available():
                adj = adj.cuda()
            
            # Forward pass
            loss = self.model.calcLosses(
                ancs, poss, negs, adj,
                attention_samples=self.attention_samples,
                handler=self.handler
            )
            
            epoch_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Backward pass
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            # Progress display - more concise
            if (i + 1) % 50 == 0 or (i + 1) == steps:
                avg_loss = sum(batch_losses[-10:]) / min(10, len(batch_losses[-10:]))
                progress = (i + 1) / steps * 100
                print(f'\r   Training: [{i+1:4d}/{steps}] ({progress:5.1f}%) | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}', 
                      end='', flush=True)
        
        print()  # New line
        avg_epoch_loss = epoch_loss / steps
        
        return avg_epoch_loss

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
                
                # Progress display
                if (i + 1) % 20 == 0 or (i + 1) == steps:
                    progress = (i + 1) / steps * 100
                    current_avg_recall = epoch_recall / num_users
                    current_avg_ndcg = epoch_ndcg / num_users
                    print(f'\r   Testing: [{i+1:3d}/{steps}] ({progress:5.1f}%) | Recall: {current_avg_recall:.4f} | NDCG: {current_avg_ndcg:.4f}', 
                          end='', flush=True)
        
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
    log(f'   Check Results/ folder for detailed outputs', level='INFO')