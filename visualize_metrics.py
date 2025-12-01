"""
Visualization script for training metrics.

Usage:
    python visualize_metrics.py

Generates plots for:
- PolicyLoss and ValueLoss over generations
- Makespan and Tardiness over generations
- Fitness (average reward) over generations
- Episode metrics distribution

Saves plots to results/plots/
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_generation_metrics(results_dir="results"):
    """Load all generation metrics from JSON files"""
    metrics_dir = os.path.join(results_dir, "metrics")
    
    generations = []
    policy_losses_avg = []
    policy_losses_std = []
    value_losses_avg = []
    value_losses_std = []
    makespans_avg = []
    makespans_std = []
    tardiness_avg = []
    returns_avg = []
    
    # Find all generation files
    gen_files = sorted([f for f in os.listdir(metrics_dir) if f.startswith("generation_")])
    
    for gen_file in gen_files:
        filepath = os.path.join(metrics_dir, gen_file)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        gen_num = data['generation']
        generations.append(gen_num)
        
        # Extract aggregate metrics
        avg_metrics = data['average_metrics']
        makespans_avg.append(avg_metrics['makespan_avg'])
        makespans_std.append(avg_metrics['makespan_std'])
        tardiness_avg.append(avg_metrics['total_tardiness_avg'])
        
        # Loss metrics (if available)
        policy_losses_avg.append(avg_metrics.get('policy_loss_avg', 0))
        policy_losses_std.append(avg_metrics.get('policy_loss_std', 0))
        value_losses_avg.append(avg_metrics.get('value_loss_avg', 0))
        value_losses_std.append(avg_metrics.get('value_loss_std', 0))
        returns_avg.append(avg_metrics.get('return_avg', 0))
    
    return {
        'generations': generations,
        'policy_losses_avg': policy_losses_avg,
        'policy_losses_std': policy_losses_std,
        'value_losses_avg': value_losses_avg,
        'value_losses_std': value_losses_std,
        'makespans_avg': makespans_avg,
        'makespans_std': makespans_std,
        'tardiness_avg': tardiness_avg,
        'returns_avg': returns_avg,
    }


def load_portfolio_fitness(results_dir="results"):
    """Load fitness from portfolio summary files"""
    portfolios_dir = os.path.join(results_dir, "portfolios")
    
    generations = []
    best_fitness = []
    avg_fitness = []
    worst_fitness = []
    
    # Find all generation summary files (not initial or final)
    gen_files = sorted([f for f in os.listdir(portfolios_dir) 
                       if f.startswith("generation_") and f.endswith("_summary.json")])
    
    for gen_file in gen_files:
        filepath = os.path.join(portfolios_dir, gen_file)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        gen_num = data['generation']
        generations.append(gen_num)
        
        # Get fitness from elite portfolios
        elite = data['elite_portfolios']
        fitnesses = [p['fitness'] for p in elite]
        
        best_fitness.append(max(fitnesses))  # Remember: reward, higher is better
        avg_fitness.append(np.mean(fitnesses))
        worst_fitness.append(min(fitnesses))
    
    return {
        'generations': generations,
        'best_fitness': best_fitness,
        'avg_fitness': avg_fitness,
        'worst_fitness': worst_fitness,
    }


def plot_metrics(results_dir="results", output_dir="results/plots"):
    """Generate all plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    metrics = load_generation_metrics(results_dir)
    fitness_data = load_portfolio_fitness(results_dir)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Figure 1: Makespan over generations
    plt.figure(figsize=(10, 6))
    gens = metrics['generations']
    plt.errorbar(gens, metrics['makespans_avg'], yerr=metrics['makespans_std'],
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Makespan', fontsize=14)
    plt.title('Makespan Improvement Over Generations', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'makespan_over_generations.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/makespan_over_generations.png")
    plt.close()
    
    # Figure 2: Tardiness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(gens, metrics['tardiness_avg'], marker='s', linewidth=2, markersize=8, color='#e74c3c')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Total Tardiness', fontsize=14)
    plt.title('Tardiness Over Generations', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tardiness_over_generations.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/tardiness_over_generations.png")
    plt.close()
    
    # Figure 3: Fitness (Best/Avg/Worst) over generations
    plt.figure(figsize=(10, 6))
    fit_gens = fitness_data['generations']
    plt.plot(fit_gens, fitness_data['best_fitness'], marker='o', linewidth=2, 
             markersize=8, label='Best', color='#2ecc71')
    plt.plot(fit_gens, fitness_data['avg_fitness'], marker='s', linewidth=2, 
             markersize=8, label='Average', color='#3498db')
    plt.plot(fit_gens, fitness_data['worst_fitness'], marker='^', linewidth=2, 
             markersize=8, label='Worst', color='#e67e22')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Fitness (Avg Reward)', fontsize=14)
    plt.title('Portfolio Fitness Evolution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fitness_evolution.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/fitness_evolution.png")
    plt.close()
    
    # Figure 4: PolicyLoss over generations
    plt.figure(figsize=(10, 6))
    plt.errorbar(gens, metrics['policy_losses_avg'], yerr=metrics['policy_losses_std'],
                 marker='D', capsize=5, capthick=2, linewidth=2, markersize=8, color='#8e44ad')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Policy Loss', fontsize=14)
    plt.title('Policy Loss Over Generations', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_loss_over_generations.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/policy_loss_over_generations.png")
    plt.close()
    
    # Figure 5: ValueLoss over generations
    plt.figure(figsize=(10, 6))
    plt.errorbar(gens, metrics['value_losses_avg'], yerr=metrics['value_losses_std'],
                 marker='v', capsize=5, capthick=2, linewidth=2, markersize=8, color='#e67e22')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Value Loss', fontsize=14)
    plt.title('Value Loss Over Generations', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'value_loss_over_generations.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/value_loss_over_generations.png")
    plt.close()
    
    # Figure 6: Combined view (3x2 subplots with losses)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    # Makespan
    axes[0, 0].errorbar(gens, metrics['makespans_avg'], yerr=metrics['makespans_std'],
                        marker='o', capsize=4, linewidth=2)
    axes[0, 0].set_xlabel('Generation', fontsize=12)
    axes[0, 0].set_ylabel('Makespan', fontsize=12)
    axes[0, 0].set_title('Makespan', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Tardiness
    axes[0, 1].plot(gens, metrics['tardiness_avg'], marker='s', linewidth=2, color='#e74c3c')
    axes[0, 1].set_xlabel('Generation', fontsize=12)
    axes[0, 1].set_ylabel('Total Tardiness', fontsize=12)
    axes[0, 1].set_title('Tardiness', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Best Fitness
    axes[1, 0].plot(fit_gens, fitness_data['best_fitness'], marker='o', linewidth=2, color='#2ecc71')
    axes[1, 0].set_xlabel('Generation', fontsize=12)
    axes[1, 0].set_ylabel('Best Fitness', fontsize=12)
    axes[1, 0].set_title('Best Portfolio Fitness', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Makespan std (variability)
    axes[1, 1].plot(gens, metrics['makespans_std'], marker='D', linewidth=2, color='#9b59b6')
    axes[1, 1].set_xlabel('Generation', fontsize=12)
    axes[1, 1].set_ylabel('Makespan Std Dev', fontsize=12)
    axes[1, 1].set_title('Solution Variability', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Policy Loss
    axes[2, 0].errorbar(gens, metrics['policy_losses_avg'], yerr=metrics['policy_losses_std'],
                       marker='D', capsize=4, linewidth=2, color='#8e44ad')
    axes[2, 0].set_xlabel('Generation', fontsize=12)
    axes[2, 0].set_ylabel('Policy Loss', fontsize=12)
    axes[2, 0].set_title('Policy Loss', fontsize=14, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Value Loss
    axes[2, 1].errorbar(gens, metrics['value_losses_avg'], yerr=metrics['value_losses_std'],
                       marker='v', capsize=4, linewidth=2, color='#e67e22')
    axes[2, 1].set_xlabel('Generation', fontsize=12)
    axes[2, 1].set_ylabel('Value Loss', fontsize=12)
    axes[2, 1].set_title('Value Loss', fontsize=14, fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics Overview', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_overview.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/metrics_overview.png")
    plt.close()
    
    print("\n" + "="*60)
    print("📊 Summary Statistics:")
    print("="*60)
    print(f"Initial Makespan: {metrics['makespans_avg'][0]:.2f} ± {metrics['makespans_std'][0]:.2f}")
    print(f"Final Makespan:   {metrics['makespans_avg'][-1]:.2f} ± {metrics['makespans_std'][-1]:.2f}")
    print(f"Improvement:      {metrics['makespans_avg'][0] - metrics['makespans_avg'][-1]:.2f} ({((metrics['makespans_avg'][0] - metrics['makespans_avg'][-1]) / metrics['makespans_avg'][0] * 100):.1f}%)")
    print()
    print(f"Initial Fitness (best): {fitness_data['best_fitness'][0]:.2f}")
    print(f"Final Fitness (best):   {fitness_data['best_fitness'][-1]:.2f}")
    print(f"Improvement:            {fitness_data['best_fitness'][-1] - fitness_data['best_fitness'][0]:.2f}")
    print("="*60)


if __name__ == "__main__":
    print("\n🎨 Generating visualization plots...")
    print("="*60)
    
    try:
        plot_metrics()
        print("\n✅ All plots generated successfully!")
        print("📁 Check results/plots/ directory")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure training has completed and metrics files exist in results/metrics/")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
