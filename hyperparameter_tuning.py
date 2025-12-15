import torch as t
import numpy as np
import json
import os
import time
from datetime import datetime
from itertools import product
import pandas as pd
from params import args
from datahandler import DataHandler
from model import TransGNN
from main import Coach, set_seed
import copy


class HyperparameterTuner:
    """
    Classe pour le tuning d'hyperparamètres de TransGNN
    Effectue une recherche en grille (Grid Search) sur 15 epochs
    """
    
    def __init__(self, dataset='yelp', base_epochs=15):
        self.dataset = dataset
        self.base_epochs = base_epochs
        self.results = []
        
        # Créer dossier pour les résultats
        self.output_dir = f'Hyperparameter_Tuning/{dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("="*80)
        print(f"🔍 HYPERPARAMETER TUNING - {dataset.upper()}")
        print(f"   Base epochs: {base_epochs}")
        print(f"   Output directory: {self.output_dir}")
        print("="*80 + "\n")
    
    def define_search_space(self):
        """
        Définir l'espace de recherche des hyperparamètres
        Vous pouvez modifier ces valeurs selon vos besoins
        """
        search_space = {
            # Learning rate - Impact majeur
            'lr': [1e-4, 5e-4, 1e-3, 5e-3],
            
            # Embedding dimension - Impact sur capacité du modèle
            'latdim': [16, 32, 64],
            
            # Attention heads - Doit diviser latdim
            'num_head': [2, 4, 8],
            
            # Attention samples (k) - Impact sur l'attention
            'k_samples': [10, 20, 30, 50],
            
            # Dropout - Régularisation
            'dropout': [0.0, 0.1, 0.2, 0.3],
            
            # Alpha - Balance structure/semantic similarity
            'alpha': [0.0, 0.3, 0.5, 0.7, 1.0],
            
            # Batch size - Impact sur training
            'batch': [2048, 4096, 8192],
            
            # Weight decay - Régularisation L2
            'decay': [0, 1e-5, 1e-4, 1e-3],
        }
        
        return search_space
    
    def get_param_combinations(self, search_space, strategy='grid', max_trials=50):
        """
        Générer les combinaisons de paramètres
        
        Args:
            search_space: dict avec les hyperparamètres
            strategy: 'grid' (exhaustif) ou 'random' (échantillonnage aléatoire)
            max_trials: nombre max d'essais pour random search
        """
        if strategy == 'grid':
            # Grid Search - toutes les combinaisons
            keys = search_space.keys()
            values = search_space.values()
            combinations = [dict(zip(keys, v)) for v in product(*values)]
            
            # Filtrer les combinaisons invalides (num_head doit diviser latdim)
            valid_combinations = []
            for combo in combinations:
                if combo['latdim'] % combo['num_head'] == 0:
                    valid_combinations.append(combo)
            
            print(f"📊 Grid Search: {len(valid_combinations)} valid combinations")
            return valid_combinations
        
        elif strategy == 'random':
            # Random Search - échantillonnage aléatoire
            combinations = []
            attempts = 0
            max_attempts = max_trials * 10
            
            while len(combinations) < max_trials and attempts < max_attempts:
                combo = {k: np.random.choice(v) for k, v in search_space.items()}
                
                # Vérifier validité
                if combo['latdim'] % combo['num_head'] == 0:
                    combinations.append(combo)
                
                attempts += 1
            
            print(f"🎲 Random Search: {len(combinations)} combinations sampled")
            return combinations
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def apply_hyperparameters(self, hyperparams):
        """Appliquer un ensemble d'hyperparamètres à args"""
        for key, value in hyperparams.items():
            setattr(args, key, value)
        
        # Forcer certains paramètres
        args.epoch = self.base_epochs
        args.data = self.dataset
        args.tstEpoch = 3  # Test tous les 3 epochs
    
    def train_single_config(self, config_id, hyperparams):
        """
        Entraîner le modèle avec une configuration d'hyperparamètres
        
        Returns:
            dict avec les résultats (recall, ndcg, training_time, etc.)
        """
        print("\n" + "="*80)
        print(f"🚀 Configuration {config_id + 1}")
        print("="*80)
        print("Hyperparameters:")
        for key, value in sorted(hyperparams.items()):
            print(f"   {key:15s} = {value}")
        print("="*80)
        
        # Appliquer les hyperparamètres
        self.apply_hyperparameters(hyperparams)
        
        # Reset seed pour reproductibilité
        set_seed(args.seed)
        
        # Charger les données (une seule fois si possible)
        if not hasattr(self, 'handler'):
            self.handler = DataHandler()
            self.handler.LoadData()
        
        try:
            # Créer et entraîner le modèle
            coach = Coach(self.handler)
            
            start_time = time.time()
            results = coach.run()
            training_time = time.time() - start_time
            
            # Extraire les métriques
            config_results = {
                'config_id': config_id,
                'best_recall': results['best_recall'],
                'best_ndcg': results['best_ndcg'],
                'best_epoch': results['best_epoch'],
                'training_time': training_time,
                'converged': True,
                **hyperparams  # Ajouter tous les hyperparamètres
            }
            
            print(f"\n✅ Config {config_id + 1} completed!")
            print(f"   Best Recall@20: {results['best_recall']:.4f}")
            print(f"   Best NDCG@20: {results['best_ndcg']:.4f}")
            print(f"   Training time: {training_time:.2f}s")
            
            return config_results
        
        except Exception as e:
            print(f"\n❌ Config {config_id + 1} failed: {str(e)}")
            
            # Retourner des résultats par défaut en cas d'échec
            return {
                'config_id': config_id,
                'best_recall': 0.0,
                'best_ndcg': 0.0,
                'best_epoch': 0,
                'training_time': 0.0,
                'converged': False,
                'error': str(e),
                **hyperparams
            }
        
        finally:
            # Nettoyer la mémoire GPU
            if t.cuda.is_available():
                t.cuda.empty_cache()
    
    def run_tuning(self, strategy='random', max_trials=20):
        """
        Exécuter le tuning complet
        
        Args:
            strategy: 'grid' ou 'random'
            max_trials: nombre max de configurations à tester (pour random)
        """
        print("\n" + "="*80)
        print("🎯 STARTING HYPERPARAMETER TUNING")
        print("="*80)
        
        # Définir l'espace de recherche
        search_space = self.define_search_space()
        
        # Générer les combinaisons
        param_combinations = self.get_param_combinations(
            search_space, 
            strategy=strategy, 
            max_trials=max_trials
        )
        
        print(f"\n📋 Total configurations to test: {len(param_combinations)}")
        print(f"⏱️  Estimated time: ~{len(param_combinations) * 5} minutes (assuming 5 min per config)\n")
        
        # Confirmation
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Tuning cancelled.")
            return
        
        # Entraîner chaque configuration
        global_start_time = time.time()
        
        for i, hyperparams in enumerate(param_combinations):
            config_results = self.train_single_config(i, hyperparams)
            self.results.append(config_results)
            
            # Sauvegarder les résultats intermédiaires
            self.save_results_incremental()
            
            # Afficher progression
            progress = (i + 1) / len(param_combinations) * 100
            elapsed = time.time() - global_start_time
            eta = elapsed / (i + 1) * (len(param_combinations) - i - 1)
            
            print(f"\n📊 Progress: {i + 1}/{len(param_combinations)} ({progress:.1f}%)")
            print(f"   Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")
        
        total_time = time.time() - global_start_time
        
        print("\n" + "="*80)
        print("✅ HYPERPARAMETER TUNING COMPLETED!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print("="*80)
        
        # Analyse finale
        self.analyze_results()
    
    def save_results_incremental(self):
        """Sauvegarder les résultats de façon incrémentale"""
        results_file = os.path.join(self.output_dir, 'results_incremental.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def analyze_results(self):
        """Analyser et visualiser les résultats du tuning"""
        print("\n" + "="*80)
        print("📊 ANALYZING RESULTS")
        print("="*80)
        
        # Convertir en DataFrame
        df = pd.DataFrame(self.results)
        
        # Filtrer les configs qui ont échoué
        df_success = df[df['converged'] == True].copy()
        
        if len(df_success) == 0:
            print("❌ No successful configurations found!")
            return
        
        print(f"\n✅ Successful configurations: {len(df_success)}/{len(df)}")
        
        # Trier par performance
        df_sorted = df_success.sort_values('best_recall', ascending=False)
        
        # Top 5 configurations
        print("\n" + "="*80)
        print("🏆 TOP 5 CONFIGURATIONS (by Recall@20)")
        print("="*80)
        
        for i, row in df_sorted.head(5).iterrows():
            print(f"\n#{row['config_id'] + 1}:")
            print(f"   Recall@20: {row['best_recall']:.4f}")
            print(f"   NDCG@20: {row['best_ndcg']:.4f}")
            print(f"   Hyperparameters:")
            for key in ['lr', 'latdim', 'num_head', 'k_samples', 'dropout', 'alpha', 'batch', 'decay']:
                if key in row:
                    print(f"      {key:12s} = {row[key]}")
        
        # Sauvegarder résultats complets
        csv_file = os.path.join(self.output_dir, 'all_results.csv')
        df_sorted.to_csv(csv_file, index=False)
        print(f"\n💾 All results saved to: {csv_file}")
        
        # Sauvegarder la meilleure config
        best_config = df_sorted.iloc[0].to_dict()
        best_config_file = os.path.join(self.output_dir, 'best_config.json')
        with open(best_config_file, 'w') as f:
            json.dump(best_config, f, indent=4)
        print(f"💾 Best configuration saved to: {best_config_file}")
        
        # Analyse de sensibilité
        self.sensitivity_analysis(df_success)
        
        # Générer visualisations
        self.plot_results(df_success)
    
    def sensitivity_analysis(self, df):
        """Analyser la sensibilité de chaque hyperparamètre"""
        print("\n" + "="*80)
        print("🔬 SENSITIVITY ANALYSIS")
        print("="*80)
        
        hyperparams = ['lr', 'latdim', 'num_head', 'k_samples', 'dropout', 'alpha', 'batch', 'decay']
        
        for param in hyperparams:
            if param not in df.columns:
                continue
            
            # Moyenne de recall pour chaque valeur du paramètre
            grouped = df.groupby(param)['best_recall'].agg(['mean', 'std', 'count'])
            
            print(f"\n{param}:")
            print(grouped.to_string())
            
            # Meilleure valeur
            best_value = grouped['mean'].idxmax()
            best_score = grouped['mean'].max()
            print(f"   → Best value: {param}={best_value} (Recall={best_score:.4f})")
    
    def plot_results(self, df):
        """Générer des visualisations des résultats"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Distribution des performances
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Recall distribution
            axes[0, 0].hist(df['best_recall'], bins=20, color='#3498DB', alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Recall@20', fontsize=12)
            axes[0, 0].set_ylabel('Frequency', fontsize=12)
            axes[0, 0].set_title('Recall@20 Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].axvline(df['best_recall'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {df["best_recall"].mean():.4f}')
            axes[0, 0].legend()
            
            # NDCG distribution
            axes[0, 1].hist(df['best_ndcg'], bins=20, color='#2ECC71', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('NDCG@20', fontsize=12)
            axes[0, 1].set_ylabel('Frequency', fontsize=12)
            axes[0, 1].set_title('NDCG@20 Distribution', fontsize=14, fontweight='bold')
            axes[0, 1].axvline(df['best_ndcg'].mean(), color='red', linestyle='--',
                              label=f'Mean: {df["best_ndcg"].mean():.4f}')
            axes[0, 1].legend()
            
            # Recall vs NDCG
            scatter = axes[1, 0].scatter(df['best_recall'], df['best_ndcg'], 
                                        c=df['training_time'], cmap='viridis', 
                                        s=100, alpha=0.6, edgecolor='black')
            axes[1, 0].set_xlabel('Recall@20', fontsize=12)
            axes[1, 0].set_ylabel('NDCG@20', fontsize=12)
            axes[1, 0].set_title('Recall vs NDCG (color = training time)', 
                                fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=axes[1, 0], label='Training Time (s)')
            
            # Top configurations
            top_5 = df.nlargest(5, 'best_recall')
            axes[1, 1].barh(range(len(top_5)), top_5['best_recall'], color='#E74C3C', alpha=0.7)
            axes[1, 1].set_yticks(range(len(top_5)))
            axes[1, 1].set_yticklabels([f"Config {i+1}" for i in top_5['config_id']])
            axes[1, 1].set_xlabel('Recall@20', fontsize=12)
            axes[1, 1].set_title('Top 5 Configurations', fontsize=14, fontweight='bold')
            axes[1, 1].invert_yaxis()
            
            # Ajouter les valeurs
            for i, (idx, row) in enumerate(top_5.iterrows()):
                axes[1, 1].text(row['best_recall'], i, f" {row['best_recall']:.4f}", 
                               va='center', fontsize=10)
            
            plt.tight_layout()
            
            plot_file = os.path.join(self.output_dir, 'tuning_results.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n📊 Visualizations saved to: {plot_file}")
            
            # 2. Heatmap de corrélation (si assez de données)
            if len(df) >= 10:
                numeric_cols = ['lr', 'latdim', 'num_head', 'k_samples', 'dropout', 
                               'alpha', 'batch', 'decay', 'best_recall', 'best_ndcg']
                numeric_cols = [col for col in numeric_cols if col in df.columns]
                
                fig, ax = plt.subplots(figsize=(12, 10))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
                ax.set_title('Hyperparameter Correlation Matrix', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                heatmap_file = os.path.join(self.output_dir, 'correlation_heatmap.png')
                plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"📊 Correlation heatmap saved to: {heatmap_file}")
        
        except Exception as e:
            print(f"⚠️  Could not generate plots: {str(e)}")


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for TransGNN')
    parser.add_argument('--dataset', type=str, default='yelp', 
                       choices=['yelp', 'gowalla', 'tmall', 'amazon-book', 'ml10m'],
                       help='Dataset name')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of epochs for each configuration')
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['grid', 'random'],
                       help='Search strategy: grid or random')
    parser.add_argument('--max_trials', type=int, default=20,
                       help='Maximum number of trials for random search')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID')
    
    args_tune = parser.parse_args()
    
    # Configurer GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args_tune.gpu
    
    # Créer le tuner
    tuner = HyperparameterTuner(
        dataset=args_tune.dataset,
        base_epochs=args_tune.epochs
    )
    
    # Lancer le tuning
    tuner.run_tuning(
        strategy=args_tune.strategy,
        max_trials=args_tune.max_trials
    )
    
    print("\n" + "="*80)
    print("🎉 TUNING COMPLETE!")
    print(f"   Results saved in: {tuner.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()