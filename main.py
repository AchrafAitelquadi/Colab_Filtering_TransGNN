import torch as t
from utils.timeLogger import log
from params import args
from model import TransGNN
from datahandler import DataHandler
import numpy as np
import os
import random
import time
import csv
from datetime import datetime


def set_seed(seed):
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
        
        log(f'TransGNN - {args.data.upper()}', level='SUCCESS')
        log(f'Users: {args.user}, Items: {args.item}, Interactions: {handler.trnLoader.dataset.__len__()}', level='INFO')
        log(f'Model: dim={args.latdim}, blocks={args.block_num}, heads={args.num_head}, k={args.k_samples}', level='INFO')
        
        self.attention_samples = None
        self.best_recall = 0
        self.best_ndcg = 0
        self.best_epoch = 0
        
        # Initialize CSV file
        self.csv_file = f'results_{args.data}_{args.save_path}.csv'
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Epoch', 'Phase', 'Loss', 'Recall@20', 'NDCG@20', 
                'Time(s)', 'Cumul_Time(s)', 'Best_Recall', 'Best_NDCG', 'Timestamp'
            ])
    
    def _log_csv(self, epoch, phase, loss, recall, ndcg, elapsed_time, cumul_time):
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, phase, 
                f'{loss:.4f}' if loss else '',
                f'{recall:.4f}' if recall else '',
                f'{ndcg:.4f}' if ndcg else '',
                f'{elapsed_time:.2f}',
                f'{cumul_time:.2f}',
                f'{self.best_recall:.4f}',
                f'{self.best_ndcg:.4f}',
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    def run(self):
        self.prepareModel()
        
        # Initialize attention samples
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
        start_time = time.time()
        
        for ep in range(args.epoch):
            # Training
            epoch_start = time.time()
            reses = self.trainEpoch()
            train_time = time.time() - epoch_start
            cumul_time = time.time() - start_time
            
            log(f'Epoch {ep}/{args.epoch} | Train Loss: {reses["Loss"]:.4f} | Time: {train_time:.1f}s', level='INFO')
            self._log_csv(ep, 'Train', reses['Loss'], None, None, train_time, cumul_time)
            
            # Testing
            if ep % args.tstEpoch == 0:
                test_start = time.time()
                reses = self.testEpoch()
                test_time = time.time() - test_start
                cumul_time = time.time() - start_time
                
                # Update best
                if reses['Recall'] > self.best_recall:
                    self.best_recall = reses['Recall']
                    self.best_ndcg = reses['NDCG']
                    self.best_epoch = ep
                    log(f'✨ New Best! Recall: {self.best_recall:.4f}, NDCG: {self.best_ndcg:.4f}', level='SUCCESS')
                
                log(f'Epoch {ep}/{args.epoch} | Test Recall: {reses["Recall"]:.4f}, NDCG: {reses["NDCG"]:.4f} | Time: {test_time:.1f}s', level='SUCCESS')
                self._log_csv(ep, 'Test', None, reses['Recall'], reses['NDCG'], test_time, cumul_time)
        
        # Final test
        reses = self.testEpoch()
        total_time = time.time() - start_time
        self._log_csv(args.epoch, 'Final', None, reses['Recall'], reses['NDCG'], 0, total_time)
        
        log('='*60, level='SUCCESS')
        log(f'Training Complete! Total time: {total_time/60:.2f} minutes', level='SUCCESS')
        log(f'Best: Recall@20={self.best_recall:.4f}, NDCG@20={self.best_ndcg:.4f} at epoch {self.best_epoch}', level='SUCCESS')
        log(f'Results saved to: {self.csv_file}', level='INFO')
        log('='*60, level='SUCCESS')
        
        return self.csv_file

    def prepareModel(self):
        self.model = TransGNN().cuda() if t.cuda.is_available() else TransGNN()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        log(f'Model ready: {total_params:,} parameters', level='SUCCESS')
    
    def trainEpoch(self):
        self.model.train()
        self.handler.trnLoader.dataset.negSampling()
        
        epLoss = 0
        steps = len(self.handler.trnLoader)
        
        for i, (ancs, poss, negs) in enumerate(self.handler.trnLoader):
            if t.cuda.is_available():
                ancs, poss, negs = ancs.long().cuda(), poss.long().cuda(), negs.long().cuda()
            else:
                ancs, poss, negs = ancs.long(), poss.long(), negs.long()
            
            adj = self.handler.torchBiAdj.cuda() if t.cuda.is_available() else self.handler.torchBiAdj
            
            loss = self.model.calcLosses(
                ancs, poss, negs, adj,
                attention_samples=self.attention_samples,
                pos_encodings=(
                    self.handler.pos_encodings['degrees'],
                    self.handler.pos_encodings['pagerank']
                )
            )
            
            epLoss += loss.item()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            # Progress bar
            if (i + 1) % 50 == 0 or (i + 1) == steps:
                print(f'\rTrain: [{i+1}/{steps}] Loss: {loss.item():.4f}', end='', flush=True)
        
        print()  # New line
        return {'Loss': epLoss / steps}

    def testEpoch(self):
        self.model.eval()
        epRecall, epNdcg = 0, 0
        num = self.handler.tstLoader.dataset.__len__()
        steps = len(self.handler.tstLoader)
        
        adj = self.handler.torchBiAdj.cuda() if t.cuda.is_available() else self.handler.torchBiAdj
        
        for i, (usr, trnMask) in enumerate(self.handler.tstLoader):
            if t.cuda.is_available():
                usr, trnMask = usr.long().cuda(), trnMask.cuda()
            else:
                usr = usr.long()
            
            usrEmbeds, itmEmbeds = self.model.predict(
                adj,
                attention_samples=self.attention_samples,
                pos_encodings=(
                    self.handler.pos_encodings['degrees'],
                    self.handler.pos_encodings['pagerank']
                )
            )
            
            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0))
            allPreds = allPreds * (1 - trnMask) - trnMask * 1e8
            
            _, topLocs = t.topk(allPreds, args.topk)
            
            recall, ndcg = self.calcRes(
                topLocs.cpu().numpy(),
                self.handler.tstLoader.dataset.tstLocs,
                usr.cpu().numpy()
            )
            epRecall += recall
            epNdcg += ndcg
            
            # Progress bar
            if (i + 1) % 20 == 0 or (i + 1) == steps:
                print(f'\rTest: [{i+1}/{steps}]', end='', flush=True)
        
        print()  # New line
        return {'Recall': epRecall / num, 'NDCG': epNdcg / num}

    def calcRes(self, topLocs, tstLocs, batIds):
        allRecall = allNdcg = 0
        
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            
            if temTstLocs is None:
                continue
            
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            
            recall = recall / tstNum if tstNum > 0 else 0
            ndcg = dcg / maxDcg if maxDcg > 0 else 0
            
            allRecall += recall
            allNdcg += ndcg
        
        return allRecall, allNdcg


if __name__ == '__main__':
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    log('='*60, level='INFO')
    log('TransGNN: Graph Neural Network + Transformer', level='SUCCESS')
    log('='*60, level='INFO')
    
    handler = DataHandler()
    handler.LoadData()
    
    coach = Coach(handler)
    csv_file = coach.run()
    
    log(f'✅ All results in: {csv_file}', level='SUCCESS')