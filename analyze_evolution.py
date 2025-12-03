"""
Deep Analysis Script: GA vs LGP Evolution

Analyze behavior across generations:
- Fitness evolution (is LGP improving?)
- Makespan/Tardiness trends
- PPO reward progression
- Action usage patterns
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_generation_data(exp_dir, num_generations=5):
    """Load metrics for all generations from an experiment"""
    gen_data = []
    
    for gen in range(1, num_generations + 1):
        data = {}
        
        # Try metrics/ directory first (LGP)
        metrics_file = f"{exp_dir}/metrics/generation_{gen}_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # LGP has nested structure
            if "average_metrics" in metrics:
                avg = metrics["average_metrics"]
                data = {
                    "generation": gen,
                    "makespan_avg": avg.get("makespan_avg"),
                    "makespan_std": avg.get("makespan_std"),
                    "makespan_min": avg.get("makespan_min"),
                    "tardiness_avg": avg.get("total_tardiness_avg"),
                    "return_avg": avg.get("return_avg"),
                    "return_std": avg.get("return_std"),
                    "policy_loss": avg.get("policy_loss_avg"),
                    "value_loss": avg.get("value_loss_avg"),
                }
        
        # Try summary file (GA)
        summary_file = f"{exp_dir}/generation_{gen}_summary.json"
        if os.path.exists(summary_file) and not data:
            # GA summary doesn't have avg metrics at top level
            # We need to load portfolio file instead
            pass
        
        # Try portfolio file for fitness info
        portfolio_file = f"{exp_dir}/portfolios/generation_{gen}_final.json"
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                portfolio_data = json.load(f)
            
            if portfolio_data.get("elite"):
                elite = portfolio_data["elite"]
                best_fitness = elite[0].get("fitness") if elite else None
                best_usage = elite[0].get("usage") if elite else 0
                
                # Count how many portfolios were actually used
                all_fitness = portfolio_data.get("all_fitness", [])
                all_usage = portfolio_data.get("all_usage", [])
                used_count = sum(1 for u in all_usage if u > 0)
                
                data.update({
                    "best_fitness": best_fitness,
                    "best_usage": best_usage,
                    "num_portfolios_used": used_count,
                    "avg_fitness": np.mean([f for f in all_fitness if f > -1e8]),  # Exclude unused
                    "elite_portfolios": len(elite),
                })
        
        if data:
            gen_data.append(data)
    
    return gen_data


def plot_evolution(ga_data, lgp_data, output_dir="experiments"):
    """Create evolution plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("GA vs LGP Evolution Analysis", fontsize=16, fontweight='bold')
    
    # Extract generations
    ga_gens = [d["generation"] for d in ga_data]
    lgp_gens = [d["generation"] for d in lgp_data]
    
    # 1. Best Fitness Evolution
    ax = axes[0, 0]
    ga_fitness = [d.get("best_fitness", 0) for d in ga_data]
    lgp_fitness = [d.get("best_fitness", 0) for d in lgp_data]
    ax.plot(ga_gens, ga_fitness, 'o-', label='GA', linewidth=2, markersize=8)
    ax.plot(lgp_gens, lgp_fitness, 's-', label='LGP', linewidth=2, markersize=8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (higher is better)")
    ax.set_title("Best Fitness Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Makespan Evolution
    ax = axes[0, 1]
    ga_makespan = [d.get("makespan_avg", 0) for d in ga_data if d.get("makespan_avg")]
    lgp_makespan = [d.get("makespan_avg", 0) for d in lgp_data if d.get("makespan_avg")]
    if ga_makespan:
        ax.plot(ga_gens[:len(ga_makespan)], ga_makespan, 'o-', label='GA', linewidth=2, markersize=8)
    if lgp_makespan:
        ax.plot(lgp_gens[:len(lgp_makespan)], lgp_makespan, 's-', label='LGP', linewidth=2, markersize=8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Makespan (lower is better)")
    ax.set_title("Makespan Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Return (Reward) Evolution
    ax = axes[0, 2]
    ga_return = [d.get("return_avg", 0) for d in ga_data if d.get("return_avg")]
    lgp_return = [d.get("return_avg", 0) for d in lgp_data if d.get("return_avg")]
    if ga_return:
        ax.plot(ga_gens[:len(ga_return)], ga_return, 'o-', label='GA', linewidth=2, markersize=8)
    if lgp_return:
        ax.plot(lgp_gens[:len(lgp_return)], lgp_return, 's-', label='LGP', linewidth=2, markersize=8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Return (higher is better)")
    ax.set_title("PPO Return Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Number of Portfolios Used
    ax = axes[1, 0]
    ga_used = [d.get("num_portfolios_used", 0) for d in ga_data]
    lgp_used = [d.get("num_portfolios_used", 0) for d in lgp_data]
    ax.plot(ga_gens, ga_used, 'o-', label='GA', linewidth=2, markersize=8)
    ax.plot(lgp_gens, lgp_used, 's-', label='LGP', linewidth=2, markersize=8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("# Portfolios Used (out of 64)")
    ax.set_title("Portfolio Usage Diversity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=64, color='gray', linestyle='--', alpha=0.5, label='Max (64)')
    
    # 5. Tardiness Evolution
    ax = axes[1, 1]
    ga_tard = [d.get("tardiness_avg", 0) for d in ga_data if d.get("tardiness_avg") is not None]
    lgp_tard = [d.get("tardiness_avg", 0) for d in lgp_data if d.get("tardiness_avg") is not None]
    if ga_tard:
        ax.plot(ga_gens[:len(ga_tard)], ga_tard, 'o-', label='GA', linewidth=2, markersize=8)
    if lgp_tard:
        ax.plot(lgp_gens[:len(lgp_tard)], lgp_tard, 's-', label='LGP', linewidth=2, markersize=8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Tardiness (lower is better)")
    ax.set_title("Tardiness Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Policy Loss Evolution
    ax = axes[1, 2]
    ga_loss = [d.get("policy_loss", 0) for d in ga_data if d.get("policy_loss") is not None]
    lgp_loss = [d.get("policy_loss", 0) for d in lgp_data if d.get("policy_loss") is not None]
    if ga_loss:
        ax.plot(ga_gens[:len(ga_loss)], ga_loss, 'o-', label='GA', linewidth=2, markersize=8)
    if lgp_loss:
        ax.plot(lgp_gens[:len(lgp_loss)], lgp_loss, 's-', label='LGP', linewidth=2, markersize=8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Policy Loss")
    ax.set_title("PPO Policy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evolution_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Evolution plots saved to: {output_dir}/evolution_analysis.png")
    plt.close()


def print_detailed_analysis(ga_data, lgp_data):
    """Print detailed text analysis"""
    print("\n" + "="*70)
    print("📊 DETAILED EVOLUTION ANALYSIS")
    print("="*70)
    
    # GA Analysis
    print("\n🔷 GA Evolution:")
    print("-" * 70)
    print(f"{'Gen':<5} {'Best Fit':<12} {'Makespan':<12} {'Return':<12} {'#Used':<8}")
    print("-" * 70)
    for d in ga_data:
        gen = d.get("generation", "?")
        fit = d.get("best_fitness", 0)
        ms = d.get("makespan_avg", "N/A")
        ret = d.get("return_avg", "N/A")
        used = d.get("num_portfolios_used", 0)
        
        ms_str = f"{ms:.1f}" if isinstance(ms, (int, float)) else str(ms)
        ret_str = f"{ret:.1f}" if isinstance(ret, (int, float)) else str(ret)
        
        print(f"{gen:<5} {fit:<12.2f} {ms_str:<12} {ret_str:<12} {used:<8}")
    
    # Check if GA is improving
    ga_fitness_trend = [d.get("best_fitness", 0) for d in ga_data]
    if len(ga_fitness_trend) >= 2:
        improvement = ga_fitness_trend[-1] - ga_fitness_trend[0]
        print(f"\n  📈 Fitness change: {ga_fitness_trend[0]:.2f} → {ga_fitness_trend[-1]:.2f} "
              f"({improvement:+.2f})")
        if improvement > 5:
            print("  ✅ GA is IMPROVING")
        elif improvement < -5:
            print("  ⚠️  GA is DECLINING")
        else:
            print("  ➡️  GA is STABLE")
    
    # LGP Analysis
    print("\n🔶 LGP Evolution:")
    print("-" * 70)
    print(f"{'Gen':<5} {'Best Fit':<12} {'Makespan':<12} {'Return':<12} {'#Used':<8}")
    print("-" * 70)
    for d in lgp_data:
        gen = d.get("generation", "?")
        fit = d.get("best_fitness", 0)
        ms = d.get("makespan_avg", "N/A")
        ret = d.get("return_avg", "N/A")
        used = d.get("num_portfolios_used", 0)
        
        ms_str = f"{ms:.1f}" if isinstance(ms, (int, float)) else str(ms)
        ret_str = f"{ret:.1f}" if isinstance(ret, (int, float)) else str(ret)
        
        print(f"{gen:<5} {fit:<12.2f} {ms_str:<12} {ret_str:<12} {used:<8}")
    
    # Check if LGP is improving
    lgp_fitness_trend = [d.get("best_fitness", 0) for d in lgp_data]
    if len(lgp_fitness_trend) >= 2:
        improvement = lgp_fitness_trend[-1] - lgp_fitness_trend[0]
        print(f"\n  📈 Fitness change: {lgp_fitness_trend[0]:.2f} → {lgp_fitness_trend[-1]:.2f} "
              f"({improvement:+.2f})")
        
        # Check if stuck
        variance = np.var(lgp_fitness_trend)
        if abs(improvement) < 5 and variance < 10:
            print("  ⚠️  LGP appears STUCK (fitness almost constant)")
            print("  💡 Recommendation: Need more features or longer training")
        elif improvement > 5:
            print("  ✅ LGP is IMPROVING")
        elif improvement < -5:
            print("  ⚠️  LGP is DECLINING")
        else:
            print("  ➡️  LGP is STABLE")
    
    # Comparison
    print("\n" + "="*70)
    print("🔍 KEY INSIGHTS")
    print("="*70)
    
    # Portfolio usage
    ga_usage = [d.get("num_portfolios_used", 0) for d in ga_data]
    lgp_usage = [d.get("num_portfolios_used", 0) for d in lgp_data]
    
    print(f"\n1. Portfolio Diversity:")
    print(f"   GA:  {np.mean(ga_usage):.1f} portfolios used on average (out of 64)")
    print(f"   LGP: {np.mean(lgp_usage):.1f} portfolios used on average (out of 64)")
    
    if np.mean(ga_usage) < 10 and np.mean(lgp_usage) < 10:
        print("   ➡️  Both methods converge to few portfolios (normal for PPO)")
    
    # Fitness progression
    print(f"\n2. Fitness Progression:")
    ga_improved = ga_fitness_trend[-1] > ga_fitness_trend[0]
    lgp_improved = lgp_fitness_trend[-1] > lgp_fitness_trend[0]
    
    if ga_improved and not lgp_improved:
        print("   ⚠️  GA improving, LGP stuck")
        print("   → LGP needs: more features, longer training, or better mutation")
    elif lgp_improved and not ga_improved:
        print("   ✅ LGP improving, GA stable")
        print("   → LGP has room to grow!")
    elif ga_improved and lgp_improved:
        print("   ✅ Both improving")
        print("   → Fair comparison, extend training to see who wins")
    else:
        print("   ➡️  Both stable")
        print("   → Quick convergence, may need stronger exploration")
    
    # Final comparison
    print(f"\n3. Final Performance:")
    final_ga_fit = ga_fitness_trend[-1]
    final_lgp_fit = lgp_fitness_trend[-1]
    
    if final_ga_fit > final_lgp_fit:
        diff = final_ga_fit - final_lgp_fit
        print(f"   GA wins by {diff:.2f} fitness points")
    elif final_lgp_fit > final_ga_fit:
        diff = final_lgp_fit - final_ga_fit
        print(f"   LGP wins by {diff:.2f} fitness points")
    else:
        print(f"   Tie!")


if __name__ == "__main__":
    print("="*70)
    print("🔬 GA vs LGP EVOLUTION ANALYZER")
    print("="*70)
    
    # Load data
    print("\n📂 Loading experiment data...")
    ga_data = load_generation_data("experiments/ga_baseline", num_generations=5)
    lgp_data = load_generation_data("experiments/lgp_test", num_generations=5)
    
    print(f"✅ Loaded {len(ga_data)} generations for GA")
    print(f"✅ Loaded {len(lgp_data)} generations for LGP")
    
    # Print detailed analysis
    print_detailed_analysis(ga_data, lgp_data)
    
    # Create plots
    print("\n📊 Generating evolution plots...")
    plot_evolution(ga_data, lgp_data, output_dir="experiments")
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70)
