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
                    <tr><td>TransGNN Blocks</td><td>{args.block_num}</td></tr>
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
    
    def finalize(self, best_results, training_time):
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
        
        # Générer les plots
        training_curves_path = self.plot_training_curves()
        comparison_path = self.plot_comparison_bars()
        
        # Générer le rapport HTML
        report_path = self.generate_summary_report(best_results, training_time)
        
        print(f"\n✅ Results saved:")
        print(f"   - Epoch results: {self.epoch_file}")
        print(f"   - Final metrics: {self.metrics_file}")
        print(f"   - Configuration: {self.config_file}")
        print(f"   - Training curves: {training_curves_path}")
        if comparison_path:
            print(f"   - Comparison plot: {comparison_path}")
        print(f"   - HTML Report: {report_path}")
        print("="*60 + "\n")
        
        return {
            'epoch_file': self.epoch_file,
            'metrics_file': self.metrics_file,
            'config_file': self.config_file,
            'training_curves': training_curves_path,
            'comparison': comparison_path,
            'report': report_path
        }