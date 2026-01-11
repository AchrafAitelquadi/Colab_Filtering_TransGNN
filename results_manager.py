import os
import csv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from params import args
import torch


class ResultsManager:
    """
    Classe pour gérer les résultats, CSV et visualisations
    """
    def __init__(self, experiment_name=None):
        self.experiment_name = experiment_name or f"{args.data}_{args.save_path}"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Créer les dossiers
        self.results_dir = 'Results'
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        self.csv_dir = os.path.join(self.results_dir, 'csv')
        self.config_dir = os.path.join(self.results_dir, 'configs')
        
        for directory in [self.results_dir, self.plots_dir, self.csv_dir, self.config_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Fichiers de sortie
        self.metrics_file = os.path.join(
            self.csv_dir, 
            f'{self.experiment_name}_{self.timestamp}_metrics.csv'
        )
        self.epoch_file = os.path.join(
            self.csv_dir, 
            f'{self.experiment_name}_{self.timestamp}_epoch_results.csv'
        )
        self.config_file = os.path.join(
            self.config_dir, 
            f'{self.experiment_name}_{self.timestamp}_config.json'
        )
        
        # Initialiser les CSV
        self._init_csv_files()
        self._save_config()
        
        # Stocker les métriques pour plotting
        self.train_losses = []
        self.train_recalls = []
        self.train_ndcgs = []
        self.test_recalls = []
        self.test_ndcgs = []
        self.epochs = []
        self.test_epochs = []
        
        # Tracking granulaire par step/batch
        self.batch_losses = []  # Loss par batch
        self.batch_steps = []   # Step global
        self.global_step = 0
    
    def _init_csv_files(self):
        """Initialiser les fichiers CSV avec headers"""
        # Fichier des métriques par epoch
        with open(self.epoch_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Epoch', 'Phase', 'Loss', 'preLoss', 'Recall@20', 'NDCG@20', 
                'Timestamp', 'Learning_Rate'
            ])
        
        # Fichier des métriques finales
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Experiment', 'Dataset', 'Final_Recall@20', 'Final_NDCG@20',
                'Best_Recall@20', 'Best_NDCG@20', 'Best_Epoch',
                'Total_Epochs', 'Training_Time', 'Timestamp'
            ])
    
    def _save_config(self):
        """Sauvegarder la configuration de l'expérience"""
        config = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'dataset': args.data,
            'model_params': {
                'latdim': args.latdim,
                'architecture': 'Trans(3) + GNN(2)',
                'num_transformer_layers': 3,
                'num_gnn_layers': 2,
                'num_head': args.num_head,
                'k_samples': args.k_samples,
                'dropout': args.dropout,
                'alpha': args.alpha,
            },
            'training_params': {
                'lr': args.lr,
                'batch_size': args.batch,
                'epochs': args.epoch,
                'weight_decay': args.decay,
            },
            'positional_encodings': {
                'use_spe': args.use_spe,
                'use_de': args.use_de,
                'use_pre': args.use_pre,
            },
            'system': {
                'gpu': args.gpu,
                'seed': args.seed,
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def log_epoch_results(self, epoch, phase, results, lr=None):
        """
        Enregistrer les résultats d'une epoch dans le CSV
        
        Args:
            epoch: numéro de l'epoch
            phase: 'Train' ou 'Test'
            results: dict avec Loss, preLoss, Recall, NDCG
            lr: learning rate actuel
        """
        with open(self.epoch_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                phase,
                results.get('Loss', ''),
                results.get('preLoss', ''),
                results.get('Recall', ''),
                results.get('NDCG', ''),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                lr if lr else ''
            ])
        
        # Stocker pour plotting
        if phase == 'Train':
            self.epochs.append(epoch)
            self.train_losses.append(results.get('Loss', 0))
            if 'Recall' in results:
                self.train_recalls.append(results['Recall'])
            if 'NDCG' in results:
                self.train_ndcgs.append(results['NDCG'])
        else:  # Test
            self.test_epochs.append(epoch)
            self.test_recalls.append(results.get('Recall', 0))
            self.test_ndcgs.append(results.get('NDCG', 0))
    
    def log_batch_step(self, loss):
        """Enregistrer la loss d'un batch individuel pour visualisation détaillée"""
        self.batch_losses.append(loss)
        self.batch_steps.append(self.global_step)
        self.global_step += 1
    
    def save_final_metrics(self, final_results, best_results, training_time):
        """
        Sauvegarder les métriques finales
        
        Args:
            final_results: résultats du test final
            best_results: dict avec best_recall, best_ndcg, best_epoch
            training_time: temps d'entraînement total en secondes
        """
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.experiment_name,
                args.data,
                final_results.get('Recall', 0),
                final_results.get('NDCG', 0),
                best_results.get('best_recall', 0),
                best_results.get('best_ndcg', 0),
                best_results.get('best_epoch', 0),
                args.epoch,
                f"{training_time:.2f}s",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
    
    def plot_training_curves(self):
        """Générer les courbes d'entraînement"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'TransGNN Training Curves - {args.data.upper()}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Training Loss
        if self.train_losses:
            axes[0, 0].plot(self.epochs, self.train_losses, 
                           linewidth=2, color='#E74C3C', marker='o', markersize=4)
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Test Recall@20
        if self.test_recalls:
            axes[0, 1].plot(self.test_epochs, self.test_recalls, 
                           linewidth=2, color='#3498DB', marker='s', markersize=6)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Recall@20', fontsize=12)
            axes[0, 1].set_title('Test Recall@20', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Marquer le meilleur
            best_idx = np.argmax(self.test_recalls)
            axes[0, 1].plot(self.test_epochs[best_idx], self.test_recalls[best_idx], 
                           'r*', markersize=15, label=f'Best: {self.test_recalls[best_idx]:.4f}')
            axes[0, 1].legend()
        
        # 3. Test NDCG@20
        if self.test_ndcgs:
            axes[1, 0].plot(self.test_epochs, self.test_ndcgs, 
                           linewidth=2, color='#2ECC71', marker='^', markersize=6)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('NDCG@20', fontsize=12)
            axes[1, 0].set_title('Test NDCG@20', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Marquer le meilleur
            best_idx = np.argmax(self.test_ndcgs)
            axes[1, 0].plot(self.test_epochs[best_idx], self.test_ndcgs[best_idx], 
                           'r*', markersize=15, label=f'Best: {self.test_ndcgs[best_idx]:.4f}')
            axes[1, 0].legend()
        
        # 4. Recall vs NDCG
        if self.test_recalls and self.test_ndcgs:
            axes[1, 1].scatter(self.test_recalls, self.test_ndcgs, 
                             c=self.test_epochs, cmap='viridis', s=100, alpha=0.6)
            axes[1, 1].set_xlabel('Recall@20', fontsize=12)
            axes[1, 1].set_ylabel('NDCG@20', fontsize=12)
            axes[1, 1].set_title('Recall vs NDCG', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
            cbar.set_label('Epoch', fontsize=10)
        
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = os.path.join(
            self.plots_dir, 
            f'{self.experiment_name}_{self.timestamp}_training_curves.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_comparison_bars(self):
        """Graphique en barres comparant les métriques finales"""
        if not self.test_recalls or not self.test_ndcgs:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Recall@20', 'NDCG@20']
        values = [
            self.test_recalls[-1] if self.test_recalls else 0,
            self.test_ndcgs[-1] if self.test_ndcgs else 0
        ]
        best_values = [
            max(self.test_recalls) if self.test_recalls else 0,
            max(self.test_ndcgs) if self.test_ndcgs else 0
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, values, width, label='Final', color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x + width/2, best_values, width, label='Best', color='#2ECC71', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'TransGNN Performance - {args.data.upper()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        plot_path = os.path.join(
            self.plots_dir, 
            f'{self.experiment_name}_{self.timestamp}_comparison.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_learning_rate_schedule(self, lr_history):
        """Visualiser l'évolution du learning rate"""
        if not lr_history:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(lr_history, linewidth=2, color='#9B59B6')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        plot_path = os.path.join(
            self.plots_dir, 
            f'{self.experiment_name}_{self.timestamp}_lr_schedule.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_batch_level_training(self):
        """Visualiser l'évolution de la loss au niveau des batches/steps"""
        if not self.batch_losses or len(self.batch_losses) < 10:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Batch-Level Training Analysis (Steps)', fontsize=16, fontweight='bold')
        
        # 1. Loss par step (toutes les valeurs)
        axes[0, 0].plot(self.batch_steps, self.batch_losses, alpha=0.3, color='gray', linewidth=0.5)
        # Moving average
        window = min(50, len(self.batch_losses) // 10)
        if window > 1:
            smoothed = np.convolve(self.batch_losses, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(self.batch_steps[window-1:], smoothed, linewidth=2, 
                          color='#E74C3C', label=f'MA-{window}')
        axes[0, 0].set_xlabel('Training Step', fontsize=12)
        axes[0, 0].set_ylabel('Batch Loss', fontsize=12)
        axes[0, 0].set_title('Loss par Step (avec lissage)', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Loss par step (zoom sur les derniers 20%)
        last_20_percent = int(len(self.batch_losses) * 0.2)
        if last_20_percent > 0:
            last_steps = self.batch_steps[-last_20_percent:]
            last_losses = self.batch_losses[-last_20_percent:]
            axes[0, 1].plot(last_steps, last_losses, alpha=0.5, color='gray', linewidth=1)
            
            # Lissage
            window_zoom = min(20, len(last_losses) // 5)
            if window_zoom > 1:
                smoothed_zoom = np.convolve(last_losses, np.ones(window_zoom)/window_zoom, mode='valid')
                axes[0, 1].plot(last_steps[window_zoom-1:], smoothed_zoom, linewidth=2.5, 
                              color='#3498DB', label=f'MA-{window_zoom}')
            axes[0, 1].set_xlabel('Training Step', fontsize=12)
            axes[0, 1].set_ylabel('Batch Loss', fontsize=12)
            axes[0, 1].set_title('Loss - Derniers 20% des Steps (Zoom)', fontsize=13, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution des losses par batch
        axes[1, 0].hist(self.batch_losses, bins=60, alpha=0.7, color='#9B59B6', edgecolor='black')
        axes[1, 0].axvline(np.mean(self.batch_losses), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(self.batch_losses):.4f}')
        axes[1, 0].axvline(np.median(self.batch_losses), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(self.batch_losses):.4f}')
        axes[1, 0].set_xlabel('Batch Loss', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Distribution des Losses par Batch', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Gradient/amélioration par step (échantillonné)
        # Prendre un échantillon pour éviter trop de calculs
        sample_rate = max(1, len(self.batch_losses) // 1000)
        sampled_losses = self.batch_losses[::sample_rate]
        sampled_steps = self.batch_steps[::sample_rate]
        
        if len(sampled_losses) > 1:
            loss_gradient = np.diff(sampled_losses)
            axes[1, 1].plot(sampled_steps[1:], loss_gradient, alpha=0.6, color='#F39C12', linewidth=1)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
            
            # Zone verte (amélioration) et rouge (dégradation)
            axes[1, 1].fill_between(sampled_steps[1:], 0, loss_gradient, 
                                    where=(loss_gradient < 0), alpha=0.3, color='green', 
                                    label='Amélioration')
            axes[1, 1].fill_between(sampled_steps[1:], 0, loss_gradient, 
                                    where=(loss_gradient >= 0), alpha=0.3, color='red', 
                                    label='Dégradation')
            
            axes[1, 1].set_xlabel('Training Step', fontsize=12)
            axes[1, 1].set_ylabel('Loss Change (Δ)', fontsize=12)
            axes[1, 1].set_title('Amélioration/Dégradation par Step', fontsize=13, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(
            self.plots_dir,
            f'{self.experiment_name}_{self.timestamp}_batch_level_training.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_embedding_analysis(self, user_embeds, item_embeds, sample_size=500):
        """Analyser les embeddings utilisateurs et items"""
        if user_embeds is None or item_embeds is None:
            return None
        
        # Convertir en numpy
        if torch.is_tensor(user_embeds):
            user_embeds = user_embeds.detach().cpu().numpy()
        if torch.is_tensor(item_embeds):
            item_embeds = item_embeds.detach().cpu().numpy()
        
        # Échantillonner pour éviter l'OOM
        n_users = min(sample_size, user_embeds.shape[0])
        n_items = min(sample_size, item_embeds.shape[0])
        
        user_sample = user_embeds[np.random.choice(user_embeds.shape[0], n_users, replace=False)]
        item_sample = item_embeds[np.random.choice(item_embeds.shape[0], n_items, replace=False)]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Embedding Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution des normes
        user_norms = np.linalg.norm(user_sample, axis=1)
        item_norms = np.linalg.norm(item_sample, axis=1)
        
        axes[0, 0].hist(user_norms, bins=50, alpha=0.7, color='#3498DB', label='Users', edgecolor='black')
        axes[0, 0].hist(item_norms, bins=50, alpha=0.7, color='#E74C3C', label='Items', edgecolor='black')
        axes[0, 0].set_xlabel('Embedding Norm', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution des Normes', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Valeurs moyennes par dimension
        user_mean = np.mean(user_sample, axis=0)
        item_mean = np.mean(item_sample, axis=0)
        
        dims = np.arange(len(user_mean))
        axes[0, 1].plot(dims, user_mean, marker='o', linewidth=1.5, color='#3498DB', label='Users', markersize=3)
        axes[0, 1].plot(dims, item_mean, marker='s', linewidth=1.5, color='#E74C3C', label='Items', markersize=3)
        axes[0, 1].set_xlabel('Dimension', fontsize=11)
        axes[0, 1].set_ylabel('Mean Value', fontsize=11)
        axes[0, 1].set_title('Valeurs Moyennes par Dimension', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Écarts-types par dimension
        user_std = np.std(user_sample, axis=0)
        item_std = np.std(item_sample, axis=0)
        
        axes[0, 2].bar(dims - 0.2, user_std, width=0.4, alpha=0.7, color='#3498DB', label='Users')
        axes[0, 2].bar(dims + 0.2, item_std, width=0.4, alpha=0.7, color='#E74C3C', label='Items')
        axes[0, 2].set_xlabel('Dimension', fontsize=11)
        axes[0, 2].set_ylabel('Std Dev', fontsize=11)
        axes[0, 2].set_title('Écart-Type par Dimension', fontsize=12, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 4. PCA visualization (2D)
        from sklearn.decomposition import PCA
        
        combined = np.vstack([user_sample, item_sample])
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined)
        
        user_reduced = reduced[:len(user_sample)]
        item_reduced = reduced[len(user_sample):]
        
        axes[1, 0].scatter(user_reduced[:, 0], user_reduced[:, 1], 
                          alpha=0.5, s=30, color='#3498DB', label='Users', edgecolors='none')
        axes[1, 0].scatter(item_reduced[:, 0], item_reduced[:, 1], 
                          alpha=0.5, s=30, color='#E74C3C', label='Items', edgecolors='none')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        axes[1, 0].set_title('PCA Visualization (2D)', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Heatmap de corrélation (users)
        sample_dims = min(20, user_sample.shape[1])
        corr_matrix = np.corrcoef(user_sample[:, :sample_dims].T)
        
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_xlabel('Dimension', fontsize=11)
        axes[1, 1].set_ylabel('Dimension', fontsize=11)
        axes[1, 1].set_title('Corrélation Dimensions (Users)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1])
        
        # 6. Distribution des valeurs d'embeddings
        axes[1, 2].hist(user_sample.flatten(), bins=60, alpha=0.7, 
                       color='#3498DB', label='Users', edgecolor='black', density=True)
        axes[1, 2].hist(item_sample.flatten(), bins=60, alpha=0.7, 
                       color='#E74C3C', label='Items', edgecolor='black', density=True)
        axes[1, 2].set_xlabel('Embedding Value', fontsize=11)
        axes[1, 2].set_ylabel('Density', fontsize=11)
        axes[1, 2].set_title('Distribution des Valeurs', fontsize=12, fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(
            self.plots_dir,
            f'{self.experiment_name}_{self.timestamp}_embedding_analysis.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_convergence_analysis(self):
        """Analyser la convergence du modèle"""
        if not self.train_losses or not self.test_recalls:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
        
        # 1. Loss avec smooth curve
        axes[0, 0].plot(self.epochs, self.train_losses, alpha=0.4, color='gray', label='Raw')
        # Moving average
        if len(self.train_losses) > 5:
            window = min(5, len(self.train_losses))
            smoothed = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(self.epochs[window-1:], smoothed, linewidth=2.5, 
                          color='#E74C3C', label='Smoothed (MA-5)')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Training Loss', fontsize=12)
        axes[0, 0].set_title('Loss Convergence', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Gradient de la loss (dérivée)
        if len(self.train_losses) > 1:
            loss_gradient = np.diff(self.train_losses)
            axes[0, 1].plot(self.epochs[1:], loss_gradient, linewidth=2, color='#9B59B6', marker='o', markersize=4)
            axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Loss Change (Δ)', fontsize=12)
            axes[0, 1].set_title('Loss Gradient (amélioration)', fontsize=13, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Taux d'amélioration Recall
        if len(self.test_recalls) > 1:
            recall_improvement = np.diff(self.test_recalls)
            epochs_test = self.test_epochs[1:]
            colors = ['green' if x > 0 else 'red' for x in recall_improvement]
            axes[1, 0].bar(epochs_test, recall_improvement, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Recall Improvement', fontsize=12)
            axes[1, 0].set_title('Recall@20 Improvement par Epoch', fontsize=13, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Stabilité (variance mobile)
        if len(self.test_recalls) > 3:
            window = min(3, len(self.test_recalls) - 1)
            rolling_std = pd.Series(self.test_recalls).rolling(window=window).std()
            axes[1, 1].plot(self.test_epochs, rolling_std, linewidth=2.5, 
                          color='#F39C12', marker='D', markersize=5)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Rolling Std (window=3)', fontsize=12)
            axes[1, 1].set_title('Stabilité des Métriques', fontsize=13, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(
            self.plots_dir,
            f'{self.experiment_name}_{self.timestamp}_convergence_analysis.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_metrics_correlation(self):
        """Analyser la corrélation entre différentes métriques"""
        if not self.test_recalls or not self.test_ndcgs or not self.train_losses:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Metrics Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Aligner les données de test avec les epochs de training
        test_losses = []
        for epoch in self.test_epochs:
            idx = self.epochs.index(epoch) if epoch in self.epochs else -1
            if idx >= 0 and idx < len(self.train_losses):
                test_losses.append(self.train_losses[idx])
            else:
                test_losses.append(None)
        
        # Filtrer les None
        valid_data = [(l, r, n) for l, r, n in zip(test_losses, self.test_recalls, self.test_ndcgs) if l is not None]
        
        if valid_data:
            test_losses_clean, recalls_clean, ndcgs_clean = zip(*valid_data)
            
            # 1. Loss vs Recall
            axes[0].scatter(test_losses_clean, recalls_clean, c=self.test_epochs[:len(recalls_clean)], 
                          cmap='viridis', s=100, alpha=0.7, edgecolors='black')
            axes[0].set_xlabel('Training Loss', fontsize=12)
            axes[0].set_ylabel('Recall@20', fontsize=12)
            axes[0].set_title('Loss vs Recall', fontsize=13, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Ajouter ligne de tendance
            z = np.polyfit(test_losses_clean, recalls_clean, 1)
            p = np.poly1d(z)
            axes[0].plot(test_losses_clean, p(test_losses_clean), "r--", alpha=0.8, linewidth=2)
            
            # 2. Loss vs NDCG
            axes[1].scatter(test_losses_clean, ndcgs_clean, c=self.test_epochs[:len(ndcgs_clean)], 
                          cmap='plasma', s=100, alpha=0.7, edgecolors='black')
            axes[1].set_xlabel('Training Loss', fontsize=12)
            axes[1].set_ylabel('NDCG@20', fontsize=12)
            axes[1].set_title('Loss vs NDCG', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Ajouter ligne de tendance
            z = np.polyfit(test_losses_clean, ndcgs_clean, 1)
            p = np.poly1d(z)
            axes[1].plot(test_losses_clean, p(test_losses_clean), "r--", alpha=0.8, linewidth=2)
        
        # 3. Heatmap de corrélation
        df_data = {
            'Recall@20': self.test_recalls,
            'NDCG@20': self.test_ndcgs
        }
        
        df = pd.DataFrame(df_data)
        corr = df.corr()
        
        im = axes[2].imshow(corr, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        axes[2].set_xticks(range(len(corr.columns)))
        axes[2].set_yticks(range(len(corr.columns)))
        axes[2].set_xticklabels(corr.columns, rotation=45, ha='right')
        axes[2].set_yticklabels(corr.columns)
        axes[2].set_title('Correlation Matrix', fontsize=13, fontweight='bold')
        
        # Ajouter les valeurs dans les cellules
        for i in range(len(corr)):
            for j in range(len(corr)):
                text = axes[2].text(j, i, f'{corr.iloc[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        
        plot_path = os.path.join(
            self.plots_dir,
            f'{self.experiment_name}_{self.timestamp}_metrics_correlation.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_performance_summary(self):
        """Vue d'ensemble complète des performances"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'TransGNN Performance Summary - {args.data.upper()}', 
                     fontsize=18, fontweight='bold')
        
        # 1. Loss over time (grand)
        ax1 = fig.add_subplot(gs[0, :])
        if self.train_losses:
            ax1.plot(self.epochs, self.train_losses, linewidth=2.5, color='#E74C3C', 
                    marker='o', markersize=4, label='Training Loss')
            ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax1.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
        
        # 2. Recall progression
        ax2 = fig.add_subplot(gs[1, 0])
        if self.test_recalls:
            ax2.plot(self.test_epochs, self.test_recalls, linewidth=2.5, color='#3498DB', 
                    marker='s', markersize=7, label='Test Recall@20')
            best_idx = np.argmax(self.test_recalls)
            ax2.plot(self.test_epochs[best_idx], self.test_recalls[best_idx], 
                    'r*', markersize=20, label=f'Best: {self.test_recalls[best_idx]:.4f}')
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Recall@20', fontsize=11)
            ax2.set_title('Recall@20', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
        
        # 3. NDCG progression
        ax3 = fig.add_subplot(gs[1, 1])
        if self.test_ndcgs:
            ax3.plot(self.test_epochs, self.test_ndcgs, linewidth=2.5, color='#2ECC71', 
                    marker='^', markersize=7, label='Test NDCG@20')
            best_idx = np.argmax(self.test_ndcgs)
            ax3.plot(self.test_epochs[best_idx], self.test_ndcgs[best_idx], 
                    'r*', markersize=20, label=f'Best: {self.test_ndcgs[best_idx]:.4f}')
            ax3.set_xlabel('Epoch', fontsize=11)
            ax3.set_ylabel('NDCG@20', fontsize=11)
            ax3.set_title('NDCG@20', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        # 4. Comparison bar chart
        ax4 = fig.add_subplot(gs[1, 2])
        if self.test_recalls and self.test_ndcgs:
            metrics = ['Recall@20', 'NDCG@20']
            best_vals = [max(self.test_recalls), max(self.test_ndcgs)]
            final_vals = [self.test_recalls[-1], self.test_ndcgs[-1]]
            
            x = np.arange(len(metrics))
            width = 0.35
            ax4.bar(x - width/2, final_vals, width, label='Final', color='#95A5A6', alpha=0.8)
            ax4.bar(x + width/2, best_vals, width, label='Best', color='#F1C40F', alpha=0.8)
            ax4.set_ylabel('Score', fontsize=11)
            ax4.set_title('Best vs Final', fontsize=12, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics, fontsize=10)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Improvement rate
        ax5 = fig.add_subplot(gs[2, 0])
        if len(self.test_recalls) > 1:
            improvements = np.diff(self.test_recalls)
            colors = ['#2ECC71' if x >= 0 else '#E74C3C' for x in improvements]
            ax5.bar(self.test_epochs[1:], improvements, color=colors, alpha=0.7, edgecolor='black')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
            ax5.set_xlabel('Epoch', fontsize=11)
            ax5.set_ylabel('Δ Recall', fontsize=11)
            ax5.set_title('Recall Improvement Rate', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Recall vs NDCG scatter
        ax6 = fig.add_subplot(gs[2, 1])
        if self.test_recalls and self.test_ndcgs:
            scatter = ax6.scatter(self.test_recalls, self.test_ndcgs, 
                                 c=self.test_epochs, cmap='coolwarm', s=120, 
                                 alpha=0.7, edgecolors='black', linewidths=1.5)
            ax6.set_xlabel('Recall@20', fontsize=11)
            ax6.set_ylabel('NDCG@20', fontsize=11)
            ax6.set_title('Recall vs NDCG', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax6)
            cbar.set_label('Epoch', fontsize=9)
        
        # 7. Statistics table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('tight')
        ax7.axis('off')
        
        if self.test_recalls and self.test_ndcgs:
            stats_data = [
                ['Metric', 'Best', 'Final', 'Mean'],
                ['Recall@20', f'{max(self.test_recalls):.4f}', 
                 f'{self.test_recalls[-1]:.4f}', f'{np.mean(self.test_recalls):.4f}'],
                ['NDCG@20', f'{max(self.test_ndcgs):.4f}', 
                 f'{self.test_ndcgs[-1]:.4f}', f'{np.mean(self.test_ndcgs):.4f}'],
                ['', '', '', ''],
                ['Training', '', '', ''],
                ['Epochs', str(len(self.epochs)), '', ''],
                ['Best Epoch', str(self.test_epochs[np.argmax(self.test_recalls)]), '', '']
            ]
            
            table = ax7.table(cellText=stats_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header
            for i in range(4):
                table[(0, i)].set_facecolor('#34495E')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        plot_path = os.path.join(
            self.plots_dir,
            f'{self.experiment_name}_{self.timestamp}_performance_summary.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_summary_report(self, best_results, training_time):
        """Générer un rapport HTML récapitulatif"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TransGNN Experiment Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2C3E50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                }}
                .section {{
                    background-color: white;
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498DB;
                    color: white;
                }}
                .metric {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2ECC71;
                }}
                img {{
                    max-width: 100%;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>TransGNN Experiment Report</h1>
                <p>Experiment: {self.experiment_name}</p>
                <p>Dataset: {args.data.upper()}</p>
                <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>Final Results</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Best Value</th>
                        <th>Best Epoch</th>
                        <th>Final Value</th>
                    </tr>
                    <tr>
                        <td>Recall@20</td>
                        <td class="metric">{best_results.get('best_recall', 0):.4f}</td>
                        <td>{best_results.get('best_epoch', 0)}</td>
                        <td>{self.test_recalls[-1] if self.test_recalls else 0:.4f}</td>
                    </tr>
                    <tr>
                        <td>NDCG@20</td>
                        <td class="metric">{best_results.get('best_ndcg', 0):.4f}</td>
                        <td>{best_results.get('best_epoch', 0)}</td>
                        <td>{self.test_ndcgs[-1] if self.test_ndcgs else 0:.4f}</td>
                    </tr>
                </table>
                <p><strong>Total Training Time:</strong> {training_time:.2f} seconds ({training_time/60:.2f} minutes)</p>
            </div>
            
            <div class="section">
                <h2>Model Configuration</h2>
                <table>
                    <tr><td>Embedding Dimension</td><td>{args.latdim}</td></tr>
                    <tr><td>Attention Heads</td><td>{args.num_head}</td></tr>
                    <tr><td>Attention Samples (k)</td><td>{args.k_samples}</td></tr>
                    <tr><td>Dropout</td><td>{args.dropout}</td></tr>
                    <tr><td>Learning Rate</td><td>{args.lr}</td></tr>
                    <tr><td>Batch Size</td><td>{args.batch}</td></tr>
                    <tr><td>Epochs</td><td>{args.epoch}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Training Curves</h2>
                <img src="../plots/{self.experiment_name}_{self.timestamp}_training_curves.png" alt="Training Curves">
            </div>
            
            <div class="section">
                <h2>Performance Comparison</h2>
                <img src="../plots/{self.experiment_name}_{self.timestamp}_comparison.png" alt="Comparison">
            </div>
        </body>
        </html>
        """
        
        report_path = os.path.join(
            self.results_dir, 
            f'{self.experiment_name}_{self.timestamp}_report.html'
        )
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def export_results_dataframe(self):
        """Exporter les résultats dans un DataFrame pandas pour analyse"""
        df = pd.read_csv(self.epoch_file)
        return df
    
    def finalize(self, best_results, training_time, user_embeds=None, item_embeds=None):
        """
        Finaliser tous les outputs (appelé à la fin de l'entraînement)
        """
        print("\n" + "="*60)
        print("Generating Results and Visualizations...")
        print("="*60)
        
        # Sauvegarder les métriques finales
        final_results = {
            'Recall': self.test_recalls[-1] if self.test_recalls else 0,
            'NDCG': self.test_ndcgs[-1] if self.test_ndcgs else 0
        }
        self.save_final_metrics(final_results, best_results, training_time)
        
        # Générer TOUS les plots
        print("   📊 Generating visualizations...")
        training_curves_path = self.plot_training_curves()
        comparison_path = self.plot_comparison_bars()
        convergence_path = self.plot_convergence_analysis()
        correlation_path = self.plot_metrics_correlation()
        summary_path = self.plot_performance_summary()
        batch_level_path = self.plot_batch_level_training()
        
        # Analyse des embeddings si disponibles
        embedding_path = None
        if user_embeds is not None and item_embeds is not None:
            print("   🔍 Analyzing embeddings...")
            embedding_path = self.plot_embedding_analysis(user_embeds, item_embeds)
        
        # Générer le rapport HTML
        report_path = self.generate_summary_report(best_results, training_time)
        
        print(f"\n✅ Results saved:")
        print(f"   - Epoch results: {self.epoch_file}")
        print(f"   - Final metrics: {self.metrics_file}")
        print(f"   - Configuration: {self.config_file}")
        print(f"\n📊 Visualizations:")
        print(f"   - Training curves: {training_curves_path}")
        if comparison_path:
            print(f"   - Comparison plot: {comparison_path}")
        if convergence_path:
            print(f"   - Convergence analysis: {convergence_path}")
        if correlation_path:
            print(f"   - Metrics correlation: {correlation_path}")
        if summary_path:
            print(f"   - Performance summary: {summary_path}")
        if batch_level_path:
            print(f"   - Batch-level training: {batch_level_path}")
        if embedding_path:
            print(f"   - Embedding analysis: {embedding_path}")
        print(f"\n📄 Report: {report_path}")
        print("="*60 + "\n")
        
        return {
            'epoch_file': self.epoch_file,
            'metrics_file': self.metrics_file,
            'config_file': self.config_file,
            'training_curves': training_curves_path,
            'comparison': comparison_path,
            'convergence': convergence_path,
            'correlation': correlation_path,
            'summary': summary_path,
            'batch_level': batch_level_path,
            'embedding_analysis': embedding_path,
            'report': report_path
        }