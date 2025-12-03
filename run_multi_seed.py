"""
Multi-Seed Experiment Runner for Statistical Validation

Runs GA vs LGP comparison across multiple random seeds to validate results.
Configuration: Round 4 (10 gens × 100 eps, mutation 0.5, full diversity mechanisms)
"""

import json
import os
import numpy as np
from datetime import datetime
from config import RANDOM_SEED, LGPConfig, CoevolutionConfig
import config

# Import from run_experiment
from run_experiment import run_experiment, compare_experiments


# Seeds for statistical validation
SEEDS = [42, 123, 999, 2025, 777]

# Configuration (Round 4 setup)
NUM_GENERATIONS = 10
EPISODES_PER_GEN = 100


def run_multi_seed_experiments(seeds=SEEDS, num_gens=NUM_GENERATIONS, eps_per_gen=EPISODES_PER_GEN):
    """
    Run experiments across multiple seeds and collect results.
    
    Returns:
        dict: Results aggregated by seed with statistics
    """
    print("="*70)
    print("🔬 MULTI-SEED STATISTICAL VALIDATION")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"  Seeds: {seeds}")
    print(f"  Generations: {num_gens}")
    print(f"  Episodes/gen: {eps_per_gen}")
    print(f"  Total experiments: {len(seeds) * 2} ({len(seeds)} seeds × 2 methods)")
    print()
    
    all_results = {
        "configuration": {
            "seeds": seeds,
            "num_generations": num_gens,
            "episodes_per_gen": eps_per_gen,
            "mutation_rate": LGPConfig.mutation_rate,
            "diversity_mechanisms": [
                "program_protection",
                "rank_based_selection",
                "best_copy_limiting"
            ]
        },
        "seeds": {}
    }
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*70}")
        print(f"🎲 SEED {seed_idx}/{len(seeds)}: {seed}")
        print(f"{'='*70}\n")
        
        # Update global seed
        config.RANDOM_SEED = seed
        
        # Create experiment directories for this seed
        ga_dir = f"experiments/multi_seed/seed_{seed}/ga"
        lgp_dir = f"experiments/multi_seed/seed_{seed}/lgp"
        
        try:
            # Run GA
            print(f"\n▶️  Running GA (seed={seed})...")
            ga_metrics = run_experiment(
                method="GA",
                output_dir=ga_dir,
                num_generations=num_gens,
                episodes_per_gen=eps_per_gen
            )
            
            # Run LGP
            print(f"\n▶️  Running LGP (seed={seed})...")
            lgp_metrics = run_experiment(
                method="LGP",
                output_dir=lgp_dir,
                num_generations=num_gens,
                episodes_per_gen=eps_per_gen
            )
            
            # Store results
            all_results["seeds"][seed] = {
                "ga": {
                    "makespan_avg": ga_metrics.get("makespan_avg"),
                    "makespan_std": ga_metrics.get("makespan_std"),
                    "makespan_min": ga_metrics.get("makespan_min"),
                    "tardiness_avg": ga_metrics.get("total_tardiness_avg"),
                    "return_avg": ga_metrics.get("return_avg"),
                    "training_time": ga_metrics.get("training_time_seconds")
                },
                "lgp": {
                    "makespan_avg": lgp_metrics.get("makespan_avg"),
                    "makespan_std": lgp_metrics.get("makespan_std"),
                    "makespan_min": lgp_metrics.get("makespan_min"),
                    "tardiness_avg": lgp_metrics.get("total_tardiness_avg"),
                    "return_avg": lgp_metrics.get("return_avg"),
                    "training_time": lgp_metrics.get("training_time_seconds"),
                    "best_fitness": lgp_metrics.get("best_portfolio", {}).get("fitness")
                },
                "comparison": {
                    "makespan_gap_pct": ((lgp_metrics.get("makespan_avg", 0) - ga_metrics.get("makespan_avg", 0)) 
                                        / ga_metrics.get("makespan_avg", 1)) * 100 if ga_metrics.get("makespan_avg") else None,
                    "lgp_wins": lgp_metrics.get("makespan_avg", float('inf')) < ga_metrics.get("makespan_avg", 0)
                }
            }
            
            print(f"\n✅ Seed {seed} complete:")
            print(f"   GA makespan:  {ga_metrics.get('makespan_avg', 'N/A'):.2f}")
            print(f"   LGP makespan: {lgp_metrics.get('makespan_avg', 'N/A'):.2f}")
            print(f"   Winner: {'🏆 LGP' if all_results['seeds'][seed]['comparison']['lgp_wins'] else '🏆 GA'}")
            
        except Exception as e:
            print(f"\n❌ Error with seed {seed}: {e}")
            all_results["seeds"][seed] = {"error": str(e)}
    
    return all_results


def compute_statistics(results):
    """Compute statistical summary across all seeds"""
    
    seeds_data = results["seeds"]
    valid_seeds = [s for s in seeds_data.values() if "error" not in s]
    
    if not valid_seeds:
        return {"error": "No valid results"}
    
    # Extract metrics
    ga_makespans = [s["ga"]["makespan_avg"] for s in valid_seeds if s["ga"]["makespan_avg"] is not None]
    lgp_makespans = [s["lgp"]["makespan_avg"] for s in valid_seeds if s["lgp"]["makespan_avg"] is not None]
    
    ga_times = [s["ga"]["training_time"] for s in valid_seeds if s["ga"]["training_time"] is not None]
    lgp_times = [s["lgp"]["training_time"] for s in valid_seeds if s["lgp"]["training_time"] is not None]
    
    gaps = [s["comparison"]["makespan_gap_pct"] for s in valid_seeds if s["comparison"]["makespan_gap_pct"] is not None]
    lgp_wins = sum(1 for s in valid_seeds if s["comparison"]["lgp_wins"])
    
    stats = {
        "num_seeds": len(valid_seeds),
        "ga": {
            "makespan_mean": np.mean(ga_makespans) if ga_makespans else None,
            "makespan_std": np.std(ga_makespans) if ga_makespans else None,
            "makespan_min": np.min(ga_makespans) if ga_makespans else None,
            "makespan_max": np.max(ga_makespans) if ga_makespans else None,
            "training_time_mean": np.mean(ga_times) if ga_times else None,
        },
        "lgp": {
            "makespan_mean": np.mean(lgp_makespans) if lgp_makespans else None,
            "makespan_std": np.std(lgp_makespans) if lgp_makespans else None,
            "makespan_min": np.min(lgp_makespans) if lgp_makespans else None,
            "makespan_max": np.max(lgp_makespans) if lgp_makespans else None,
            "training_time_mean": np.mean(lgp_times) if lgp_times else None,
        },
        "comparison": {
            "gap_mean_pct": np.mean(gaps) if gaps else None,
            "gap_std_pct": np.std(gaps) if gaps else None,
            "lgp_win_rate": (lgp_wins / len(valid_seeds)) * 100 if valid_seeds else 0,
            "lgp_wins": lgp_wins,
            "ga_wins": len(valid_seeds) - lgp_wins
        }
    }
    
    return stats


def print_summary(results, stats):
    """Print formatted summary of multi-seed results"""
    
    print("\n" + "="*70)
    print("📊 STATISTICAL SUMMARY")
    print("="*70)
    
    print(f"\n📈 Aggregated Results ({stats['num_seeds']} seeds):")
    print(f"\n{'Metric':<30} {'GA':<20} {'LGP':<20} {'Winner'}")
    print("-"*70)
    
    # Makespan
    ga_ms = stats["ga"]["makespan_mean"]
    lgp_ms = stats["lgp"]["makespan_mean"]
    gap = stats["comparison"]["gap_mean_pct"]
    
    print(f"{'Makespan (mean ± std)':<30} "
          f"{ga_ms:.2f} ± {stats['ga']['makespan_std']:.2f}   "
          f"{lgp_ms:.2f} ± {stats['lgp']['makespan_std']:.2f}   "
          f"{'🏆 LGP' if lgp_ms < ga_ms else '🏆 GA'}")
    
    print(f"{'Makespan (range)':<30} "
          f"[{stats['ga']['makespan_min']:.2f}, {stats['ga']['makespan_max']:.2f}]   "
          f"[{stats['lgp']['makespan_min']:.2f}, {stats['lgp']['makespan_max']:.2f}]")
    
    # Training time
    print(f"{'Training time (mean)':<30} "
          f"{stats['ga']['training_time_mean']:.1f}s   "
          f"{stats['lgp']['training_time_mean']:.1f}s   "
          f"{'🏆 LGP' if stats['lgp']['training_time_mean'] < stats['ga']['training_time_mean'] else '🏆 GA'}")
    
    print("\n" + "-"*70)
    print(f"\n🎯 Win/Loss Record:")
    print(f"   LGP wins: {stats['comparison']['lgp_wins']}/{stats['num_seeds']} "
          f"({stats['comparison']['lgp_win_rate']:.1f}%)")
    print(f"   GA wins:  {stats['comparison']['ga_wins']}/{stats['num_seeds']} "
          f"({100 - stats['comparison']['lgp_win_rate']:.1f}%)")
    
    print(f"\n📊 Average Gap: {gap:+.2f}% " 
          f"({'LGP better' if gap < 0 else 'GA better'})")
    print(f"   Gap std: ±{stats['comparison']['gap_std_pct']:.2f}%")
    
    # Statistical significance (simple t-test approximation)
    if stats['num_seeds'] >= 3:
        import scipy.stats as stats_scipy
        from scipy.stats import ttest_ind
        
        ga_makespans = [results["seeds"][s]["ga"]["makespan_avg"] 
                       for s in results["seeds"] if "error" not in results["seeds"][s]]
        lgp_makespans = [results["seeds"][s]["lgp"]["makespan_avg"] 
                        for s in results["seeds"] if "error" not in results["seeds"][s]]
        
        if len(ga_makespans) == len(lgp_makespans) and len(ga_makespans) >= 2:
            t_stat, p_value = ttest_ind(ga_makespans, lgp_makespans)
            
            print(f"\n🔬 Statistical Test (t-test):")
            print(f"   t-statistic: {t_stat:.3f}")
            print(f"   p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"   ✅ Statistically significant (p < 0.05)")
                print(f"      {'LGP' if lgp_ms < ga_ms else 'GA'} is significantly better")
            else:
                print(f"   ⚠️  Not statistically significant (p >= 0.05)")
                print(f"      Results could be due to random variation")


def save_results(results, stats, output_file="experiments/multi_seed/results.json"):
    """Save all results to JSON"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": results["configuration"],
        "seeds": results["seeds"],
        "statistics": stats
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    print("\n🚀 Starting multi-seed validation...")
    print(f"⏱️  Estimated time: ~{len(SEEDS) * 1.5} minutes\n")
    
    # Run experiments
    results = run_multi_seed_experiments()
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Print summary
    print_summary(results, stats)
    
    # Save results
    save_results(results, stats)
    
    print("\n" + "="*70)
    print("✅ Multi-seed validation complete!")
    print("="*70)
