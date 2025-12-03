"""
Experiment Script: GA vs LGP Comparison

Run both GA and LGP on same dataset and compare results.
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim

# Imports
from config import (
    PPOConfig, CoevolutionConfig, LGPConfig, EnvironmentConfig,
    RANDOM_SEED, print_config_summary
)
from main import (
    DynamicSchedulingEnv, PPOActorCritic,
    select_action, compute_returns,
    initialize_lgp_action_library
)
from coevolution_trainer import train_with_coevolution, CoevolutionConfig as CoevoCfg
from lgp_coevolution import train_with_coevolution_lgp
from lgp_generator import LGPGenerator


def run_experiment(method="GA", exp_name="exp_ga", num_generations=5, episodes_per_gen=100):
    """
    Run single experiment with GA or LGP
    
    Args:
        method: "GA" or "LGP"
        exp_name: experiment name for output directory
        num_generations: number of generations to train
        episodes_per_gen: episodes per generation
    """
    print("\n" + "="*70)
    print(f"🧪 EXPERIMENT: {method} - {exp_name}")
    print("="*70)
    
    output_dir = f"experiments/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save experiment config
    exp_config = {
        "method": method,
        "timestamp": datetime.now().isoformat(),
        "num_generations": num_generations,
        "episodes_per_gen": episodes_per_gen,
        "pool_size": LGPConfig.pool_size,
        "elite_size": CoevolutionConfig.elite_size,
        "n_replace": CoevolutionConfig.n_replace,
        "random_seed": RANDOM_SEED,
    }
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    # Initialize
    lr = PPOConfig.learning_rate
    gamma = PPOConfig.gamma
    ppo_epochs = PPOConfig.ppo_epochs
    clip_epsilon = PPOConfig.clip_epsilon
    entropy_coef = PPOConfig.entropy_coef
    
    # Create config
    cfg = CoevoCfg(
        num_generations=num_generations,
        episodes_per_gen=episodes_per_gen,
        max_steps_per_episode=CoevolutionConfig.max_steps_per_episode,
        gamma=gamma,
        ppo_epochs=ppo_epochs,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        elite_size=CoevolutionConfig.elite_size,
        n_replace=CoevolutionConfig.n_replace,
        warmup_episodes=CoevolutionConfig.warmup_episodes,
        mutation_sigma=CoevolutionConfig.mutation_sigma,
        dr_mutation_prob=CoevolutionConfig.dr_mutation_prob,
        mh_name_mutation_prob=CoevolutionConfig.mh_name_mutation_prob
    )
    
    start_time = time.time()
    
    if method == "GA":
        # GA: Use existing portfolio evolution
        action_library = initialize_lgp_action_library(
            pool_size=LGPConfig.pool_size,
            dr_list=LGPConfig.available_dr,
            mh_list=LGPConfig.available_mh,
            seed=RANDOM_SEED
        )
        
        env = DynamicSchedulingEnv(
            lambda_tardiness=EnvironmentConfig.lambda_tardiness,
            action_library=action_library,
            action_budget_s=LGPConfig.action_budget_s
        )
        
        model = PPOActorCritic(env.observation_space.shape[0], env.action_space.n)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_with_coevolution(
            env=env,
            action_library=action_library,
            model=model,
            optimizer=optimizer,
            select_action_fn=select_action,
            compute_returns_fn=compute_returns,
            cfg=cfg,
            output_dir=output_dir
        )
        
        final_action_library = action_library
        
    elif method == "LGP":
        # LGP: Use program evolution
        import random as _random
        lgp_rng = _random.Random(RANDOM_SEED)
        lgp_gen = LGPGenerator(
            max_length=LGPConfig.max_program_length,
            min_length=LGPConfig.min_program_length,
            num_registers=LGPConfig.num_registers,
            rng=lgp_rng,
        )
        lgp_programs = [
            lgp_gen.generate_random_program()
            for _ in range(LGPConfig.pool_size)
        ]
        
        # Placeholder action library for env initialization
        action_library = initialize_lgp_action_library(
            pool_size=LGPConfig.pool_size,
            dr_list=LGPConfig.available_dr,
            mh_list=LGPConfig.available_mh,
            seed=RANDOM_SEED
        )
        
        env = DynamicSchedulingEnv(
            lambda_tardiness=EnvironmentConfig.lambda_tardiness,
            action_library=action_library,
            action_budget_s=LGPConfig.action_budget_s
        )
        
        model = PPOActorCritic(env.observation_space.shape[0], env.action_space.n)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        lgp_programs, final_action_library = train_with_coevolution_lgp(
            env=env,
            lgp_programs=lgp_programs,
            model=model,
            optimizer=optimizer,
            select_action_fn=select_action,
            compute_returns_fn=compute_returns,
            cfg=cfg,
            output_dir=output_dir
        )
        
        # Save LGP programs
        programs_data = {
            "num_programs": len(lgp_programs),
            "programs": [prog.to_dict() for prog in lgp_programs]
        }
        with open(f"{output_dir}/lgp_programs_final.json", 'w') as f:
            json.dump(programs_data, f, indent=2)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed = time.time() - start_time
    
    # Save model
    torch.save(model.state_dict(), f"{output_dir}/model_final.pth")
    
    # Collect final metrics
    final_metrics = collect_final_metrics(output_dir, num_generations)
    final_metrics["training_time_seconds"] = elapsed
    final_metrics["method"] = method
    
    with open(f"{output_dir}/final_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n✅ {method} Experiment Complete!")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Output: {output_dir}/")
    
    return final_metrics


def collect_final_metrics(output_dir, num_generations):
    """Extract key metrics from experiment results"""
    metrics = {}
    
    # LEARNING CURVES: Track evolution across generations
    avg_rewards_per_gen = []
    best_rewards_per_gen = []
    makespans_per_gen = []
    
    # Load metrics for all generations
    for gen in range(1, num_generations + 1):
        try:
            metrics_file = f"{output_dir}/metrics/generation_{gen}_metrics.json"
            with open(metrics_file, 'r') as f:
                gen_data = json.load(f)
            
            # Track makespan evolution
            makespans_per_gen.append(gen_data.get("makespan_avg", None))
            
            # Track reward evolution (if available)
            if "return_avg" in gen_data:
                avg_rewards_per_gen.append(gen_data["return_avg"])
                best_rewards_per_gen.append(gen_data.get("return_max", gen_data["return_avg"]))
        except Exception as e:
            print(f"  ⚠️ Could not load generation {gen} metrics: {e}")
            makespans_per_gen.append(None)
    
   # If no metrics files found, try alternative location (GA uses different path)
    if all(m is None for m in makespans_per_gen):
        print(f"  ℹ️  Trying alternative metrics path...")
        # GA might save to different location, check summary files
        for gen in range(1, num_generations + 1):
            try:
                summary_file = f"{output_dir}/generation_{gen}_summary.json"
                with open(summary_file, 'r') as f:
                    gen_data = json.load(f)
                
                makespans_per_gen[gen-1] = gen_data.get("makespan_avg", None)
                if "return_avg" in gen_data:
                    avg_rewards_per_gen.append(gen_data["return_avg"])
                    best_rewards_per_gen.append(gen_data.get("return_max", gen_data["return_avg"]))
            except:
                pass
    
    # Load FINAL generation metrics for summary
    try:
        metrics_file = f"{output_dir}/metrics/generation_{num_generations}_metrics.json"
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        # LGP saves in nested structure: average_metrics
        if "average_metrics" in metrics_data:
            final_gen_data = metrics_data["average_metrics"]
        else:
            final_gen_data = metrics_data
        
        metrics["makespan_avg"] = final_gen_data.get("makespan_avg")
        metrics["makespan_std"] = final_gen_data.get("makespan_std")
        metrics["makespan_min"] = final_gen_data.get("makespan_min")
        metrics["total_tardiness_avg"] = final_gen_data.get("total_tardiness_avg")
        metrics["policy_loss_avg"] = final_gen_data.get("policy_loss_avg", None)
        metrics["value_loss_avg"] = final_gen_data.get("value_loss_avg", None)
        metrics["return_avg"] = final_gen_data.get("return_avg", None)
        metrics["return_max"] = final_gen_data.get("return_max", None)
    except Exception as e:
        print(f"  ⚠️ Could not load final generation metrics from metrics/: {e}")
        
        # Try summary file as fallback (GA uses this)
        try:
            summary_file = f"{output_dir}/generation_{num_generations}_summary.json"
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # GA summary has elite_portfolios but no makespan avg at top level
            # We need to compute from portfolio data
            metrics["makespan_avg"] = None  # Not available in GA summary
            metrics["makespan_std"] = None
            metrics["makespan_min"] = None
            metrics["total_tardiness_avg"] = None
            metrics["policy_loss_avg"] = None
            metrics["value_loss_avg"] = None
            metrics["return_avg"] = None
            metrics["return_max"] = None
            print(f"  ℹ️  Loaded GA summary (partial metrics only)")
        except Exception as e2:
            print(f"  ⚠️ Could not load from summary either: {e2}")
            metrics = {
                "makespan_avg": None,
                "makespan_std": None,
                "makespan_min": None,
                "total_tardiness_avg": None,
                "policy_loss_avg": None,
                "value_loss_avg": None,
                "return_avg": None,
                "return_max": None,
            }
    
    # Add learning curves
    metrics["makespans_per_generation"] = makespans_per_gen
    metrics["avg_rewards_per_generation"] = avg_rewards_per_gen if avg_rewards_per_gen else None
    metrics["best_rewards_per_generation"] = best_rewards_per_gen if best_rewards_per_gen else None
    
    # Load portfolio data
    try:
        portfolio_file = f"{output_dir}/portfolios/generation_{num_generations}_final.json"
        with open(portfolio_file, 'r') as f:
            portfolio_data = json.load(f)
        
        # Best portfolio
        if portfolio_data.get("elite"):
            best = portfolio_data["elite"][0]
            metrics["best_portfolio"] = {
                "dr": best["dr"]["name"],
                "mh_genes": [{"name": mh["name"], "weight": mh["weight_norm"]} 
                            for mh in best["mh_genes"]],
                "fitness": best.get("fitness"),
                "usage": best.get("usage")
            }
    except Exception as e:
        print(f"  ⚠️ Could not load portfolio data: {e}")
        metrics["best_portfolio"] = None
    
    print(f"  ✅ Metrics collected: makespan_avg={metrics.get('makespan_avg')}, "
          f"return_avg={metrics.get('return_avg')}")
    
    return metrics


def compare_experiments(exp_ga, exp_lgp):
    """Compare GA vs LGP results"""
    print("\n" + "="*70)
    print("📊 COMPARISON: GA vs LGP")
    print("="*70)
    
    # Load metrics
    with open(f"experiments/{exp_ga}/final_metrics.json", 'r') as f:
        ga_metrics = json.load(f)
    
    with open(f"experiments/{exp_lgp}/final_metrics.json", 'r') as f:
        lgp_metrics = json.load(f)
    
    comparison = {
        "ga": ga_metrics,
        "lgp": lgp_metrics,
        "comparison_timestamp": datetime.now().isoformat()
    }
    
    # Print comparison
    print("\n📈 Performance Metrics:")
    print(f"{'Metric':<30} {'GA':<15} {'LGP':<15} {'Δ%':<10}")
    print("-" * 70)
    
    def compare_metric(name, ga_val, lgp_val, lower_is_better=True):
        if ga_val is None or lgp_val is None:
            delta_pct = "N/A"
            winner = ""
        elif abs(ga_val) < 1e-6:  # Handle zero or very small values
            if abs(lgp_val) < 1e-6:
                delta_pct = "0.0%"
                winner = "🟰 Tie"
            else:
                delta_pct = "N/A"  # Can't compute % change from zero
                winner = "🏆 LGP" if lgp_val < ga_val else "🏆 GA"
        else:
            delta = ((lgp_val - ga_val) / ga_val) * 100
            delta_pct = f"{delta:+.1f}%"
            
            if lower_is_better:
                winner = "🏆 LGP" if lgp_val < ga_val else "🏆 GA"
            else:
                winner = "🏆 LGP" if lgp_val > ga_val else "🏆 GA"
        
        print(f"{name:<30} {str(ga_val):<15} {str(lgp_val):<15} {delta_pct:<10} {winner}")

    
    compare_metric("Makespan (avg)", 
                   ga_metrics.get("makespan_avg"), 
                   lgp_metrics.get("makespan_avg"),
                   lower_is_better=True)
    
    compare_metric("Makespan (min)", 
                   ga_metrics.get("makespan_min"), 
                   lgp_metrics.get("makespan_min"),
                   lower_is_better=True)
    
    compare_metric("Tardiness (avg)", 
                   ga_metrics.get("total_tardiness_avg"), 
                   lgp_metrics.get("total_tardiness_avg"),
                   lower_is_better=True)
    
    compare_metric("Training Time (s)", 
                   ga_metrics.get("training_time_seconds"), 
                   lgp_metrics.get("training_time_seconds"),
                   lower_is_better=True)
    
    # Best portfolios
    print("\n🏆 Best Portfolios:")
    print("\nGA Best Portfolio:")
    if ga_metrics.get("best_portfolio"):
        bp = ga_metrics["best_portfolio"]
        print(f"  DR: {bp['dr']}")
        for mh in bp['mh_genes']:
            print(f"  MH: {mh['name']} (weight={mh['weight']:.3f})")
        print(f"  Fitness: {bp.get('fitness')}")
        print(f"  Usage: {bp.get('usage')}")
    
    print("\nLGP Best Portfolio:")
    if lgp_metrics.get("best_portfolio"):
        bp = lgp_metrics["best_portfolio"]
        print(f"  DR: {bp['dr']}")
        for mh in bp['mh_genes']:
            print(f"  MH: {mh['name']} (weight={mh['weight']:.3f})")
        print(f"  Fitness: {bp.get('fitness')}")
        print(f"  Usage: {bp.get('usage')}")
    
    # Save comparison
    with open("experiments/comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Comparison saved to: experiments/comparison.json")
    print("="*70)
    
    return comparison


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 GA vs LGP EXPERIMENT SUITE")
    print("="*70)
    
    # Configuration
    NUM_GENERATIONS = 10  # Increased from 5 for fair comparison
    EPISODES_PER_GEN = 100  # Increased from 50 to give LGP time to evolve
    
    print(f"\n⚙️  Config: {NUM_GENERATIONS} generations × {EPISODES_PER_GEN} episodes")
    print(f"   Pool size: {LGPConfig.pool_size}")
    print(f"   Random seed: {RANDOM_SEED}")
    
    # Run GA
    print("\n" + "="*70)
    print("Phase 1: Running GA Baseline")
    print("="*70)
    ga_metrics = run_experiment(
        method="GA",
        exp_name="ga_baseline",
        num_generations=NUM_GENERATIONS,
        episodes_per_gen=EPISODES_PER_GEN
    )
    
    # Run LGP
    print("\n" + "="*70)
    print("Phase 2: Running LGP")
    print("="*70)
    lgp_metrics = run_experiment(
        method="LGP",
        exp_name="lgp_test",
        num_generations=NUM_GENERATIONS,
        episodes_per_gen=EPISODES_PER_GEN
    )
    
    # Compare
    comparison = compare_experiments("ga_baseline", "lgp_test")
    
    print("\n✅ Experiment suite complete!")
    print("📁 Results in experiments/ directory")
