# coevolution_trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import json
import os
from pathlib import Path

from lgp_actions import ActionIndividual, describe_individual
from dispatching_registry import DR_REGISTRY
from mh_registry import MH_REGISTRY


# ==================== Portfolio Logging Helper ====================

def log_top_k_portfolios(action_library: List[ActionIndividual], 
                        fitnesses: np.ndarray, 
                        k: int = 3, 
                        maximize: bool = True):
    """
    Log top k portfolios dựa trên fitness.
    maximize=True: fitness càng lớn càng tốt (reward)
    maximize=False: fitness càng nhỏ càng tốt (cost/makespan)
    """
    if maximize:
        idxs = fitnesses.argsort()[::-1][:k]  # sort giảm dần
    else:
        idxs = fitnesses.argsort()[:k]        # sort tăng dần

    print(f"\n🏆 Top {k} portfolios this generation:")
    for rank, idx in enumerate(idxs, start=1):
        ind = action_library[int(idx)]
        print(f"  #{rank} (idx={idx}, fitness={fitnesses[idx]:.2f}): {describe_individual(ind)}")



# ==================== Helper Functions for Portfolio Tracking ====================

def portfolio_to_dict(individual: ActionIndividual, index: int = None, 
                     fitness: float = None, usage: int = None) -> Dict[str, Any]:
    """Convert ActionIndividual to dictionary format with normalized weights"""
    # Import here to avoid circular dependency
    from lgp_actions import individual_normalized_weights
    
    norm_ws = individual_normalized_weights(individual)
    
    portfolio = {
        "dr": {
            "name": individual.genes[0].name,
            "weight": round(individual.genes[0].w_raw, 4)
        },
        "mh_genes": [
            {
                "name": g.name,
                "weight_raw": round(g.w_raw, 4),
                "weight_norm": round(w_norm, 4)
            }
            for g, w_norm in zip(individual.genes[1:], norm_ws)
        ]
    }
    
    if index is not None:
        portfolio["index"] = index
    if fitness is not None:
        portfolio["fitness"] = round(float(fitness), 2)
    if usage is not None:
        portfolio["usage"] = int(usage)
    
    return portfolio


def save_portfolios_json(portfolios_data: Dict[str, Any], filename: str):
    """Save portfolios to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(portfolios_data, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved to: {filename}")


def print_portfolio(individual: ActionIndividual, index: int = None, 
                   fitness: float = None, usage: int = None, prefix: str = ""):
    """Pretty print a single portfolio"""
    dr_name = individual.genes[0].name
    mh_str = " ".join([f"{g.name}({g.w_raw:.2f})" for g in individual.genes[1:]])
    
    line = f"{prefix}"
    if index is not None:
        line += f"#{index:2d}: "
    if fitness is not None:
        line += f"fit={fitness:7.2f} "
    if usage is not None:
        line += f"use={usage:3d} "
    line += f"[{dr_name} → {mh_str}]"
    
    print(line)


def print_portfolios_summary(action_library: List[ActionIndividual],
                            elite_indices: List[int],
                            loser_indices: List[int],
                            fitness: np.ndarray,
                            usage: np.ndarray,
                            parent_pairs: List[tuple] = None):
    """Print detailed summary of portfolios for current generation"""
    
    print(f"\n📊 Elite Portfolios (Top {len(elite_indices)}):")
    for idx in elite_indices[:5]:  # Show top 5
        print_portfolio(action_library[idx], idx, fitness[idx], usage[idx], "  ")
    if len(elite_indices) > 5:
        print(f"  ... and {len(elite_indices) - 5} more elite portfolios")
    
    print(f"\n❌ Replaced Portfolios ({len(loser_indices)} losers):")
    for idx in loser_indices:
        print_portfolio(action_library[idx], idx, fitness[idx], usage[idx], "  ")
    
    if parent_pairs:
        print(f"\n✨ New Portfolios (after evolution):")
        for i, idx in enumerate(loser_indices):
            pa_idx, pb_idx = parent_pairs[i]
            print(f"  #{idx:2d}: parents=[{pa_idx},{pb_idx}] ", end="")
            mh_str = " ".join([f"{g.name}({g.w_raw:.2f})" for g in action_library[idx].genes[1:]])
            print(f"[{action_library[idx].genes[0].name} → {mh_str}]")


def collect_episode_metrics(env) -> Dict[str, Any]:
    """Collect metrics from environment after episode"""
    metrics = env.get_metrics()
    return {
        "makespan": round(float(metrics["makespan"]), 2),
        "tardiness_normal": round(float(metrics["tardiness_normal"]), 2),
        "tardiness_urgent": round(float(metrics["tardiness_urgent"]), 2),
        "total_tardiness": round(float(metrics["tardiness_normal"] + metrics["tardiness_urgent"]), 2)
    }


def save_generation_metrics(generation: int, 
                           episodes_metrics: List[Dict[str, Any]],
                           output_dir: str):
    """Save aggregated metrics for a generation"""
    # Calculate averages
    avg_metrics = {
        "makespan_avg": np.mean([m["makespan"] for m in episodes_metrics]),
        "makespan_std": np.std([m["makespan"] for m in episodes_metrics]),
        "makespan_min": np.min([m["makespan"] for m in episodes_metrics]),
        "makespan_max": np.max([m["makespan"] for m in episodes_metrics]),
        "tardiness_normal_avg": np.mean([m["tardiness_normal"] for m in episodes_metrics]),
        "tardiness_urgent_avg": np.mean([m["tardiness_urgent"] for m in episodes_metrics]),
        "total_tardiness_avg": np.mean([m["total_tardiness"] for m in episodes_metrics]),
        "policy_loss_avg": np.mean([m["policy_loss"] for m in episodes_metrics]),
        "policy_loss_std": np.std([m["policy_loss"] for m in episodes_metrics]),
        "value_loss_avg": np.mean([m["value_loss"] for m in episodes_metrics]),
        "value_loss_std": np.std([m["value_loss"] for m in episodes_metrics]),
        "return_avg": np.mean([m["return"] for m in episodes_metrics]),
        "return_std": np.std([m["return"] for m in episodes_metrics]),
    }
    
    gen_data = {
        "generation": generation,
        "num_episodes": len(episodes_metrics),
        "average_metrics": {k: round(float(v), 4) for k, v in avg_metrics.items()},
        "all_episodes": episodes_metrics
    }
    
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    filename = os.path.join(metrics_dir, f"generation_{generation}_metrics.json")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(gen_data, f, indent=2, ensure_ascii=False)
    
    print(f"  📊 Metrics: Makespan={avg_metrics['makespan_avg']:.2f}±{avg_metrics['makespan_std']:.2f}, "
          f"Tardiness={avg_metrics['total_tardiness_avg']:.2f}, "
          f"PolicyLoss={avg_metrics['policy_loss_avg']:.2f}, ValueLoss={avg_metrics['value_loss_avg']:.2f}")


def save_final_results(env, output_dir: str, generation: int):
    """Save final schedule and metrics"""
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    # Get final metrics
    metrics = collect_episode_metrics(env)
    
    # Save schedule events
    schedule_data = {
        "generation": generation,
        "metrics": metrics,
        "schedule_events": [
            {
                "job": str(e["job"]),
                "op_index": int(e["op_index"]),
                "machine": int(e["machine"]),
                "start": round(float(e["start"]), 2),
                "finish": round(float(e["finish"]), 2),
                "op_id": int(e["op_id"])
            }
            for e in env.current_schedule_events
        ]
    }
    
    schedule_file = os.path.join(final_dir, "final_schedule.json")
    with open(schedule_file, 'w', encoding='utf-8') as f:
        json.dump(schedule_data, f, indent=2, ensure_ascii=False)
    
    # Save metrics summary
    metrics_file = os.path.join(final_dir, "final_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n  💾 Final schedule saved to: {schedule_file}")
    print(f"  💾 Final metrics saved to: {metrics_file}")
    print(f"  📊 Makespan: {metrics['makespan']}, Total Tardiness: {metrics['total_tardiness']}")
    
    # Optional: Generate Gantt chart
    try:
        from main import plot_gantt
        import matplotlib.pyplot as plt
        
        plt.figure()
        plot_gantt(env.current_schedule_events)
        gantt_file = os.path.join(final_dir, "final_gantt.png")
        plt.savefig(gantt_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 Gantt chart saved to: {gantt_file}")
    except Exception as e:
        print(f"  ⚠️ Could not generate Gantt chart: {e}")



@dataclass
class CoevolutionConfig:
    num_generations: int = 20
    episodes_per_gen: int = 10
    max_steps_per_episode: int = 200
    gamma: float = 0.9
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    elite_size: int = 16
    n_replace: int = 4
    warmup_episodes: int = 2
    mutation_sigma: float = 0.3
    dr_mutation_prob: float = 0.1
    mh_name_mutation_prob: float = 0.2


def _select_elite_indices(fitness: np.ndarray, elite_size: int) -> List[int]:
    sorted_idx = np.argsort(-fitness)
    return sorted_idx[:elite_size].tolist()


def _sample_parent_indices(fitness: np.ndarray,
                           rng: np.random.Generator,
                           k: int) -> List[int]:
    if np.all(fitness == fitness[0]):
        probs = np.ones_like(fitness) / len(fitness)
    else:
        scaled = fitness - np.max(fitness)
        exp = np.exp(scaled)
        probs = exp / np.sum(exp)
    return rng.choice(len(fitness), size=k, replace=True, p=probs).tolist()


def _clone_individual(ind: ActionIndividual) -> ActionIndividual:
    from lgp_actions import Gene
    new_genes = [Gene(g.kind, g.name, g.w_raw) for g in ind.genes]
    return ActionIndividual(genes=new_genes)


def _mutate_individual(ind: ActionIndividual,
                       rng: np.random.Generator,
                       cfg: CoevolutionConfig) -> ActionIndividual:
    child = _clone_individual(ind)

    dr_names = list(DR_REGISTRY.keys())
    mh_names = list(MH_REGISTRY.keys())

    # mutate DR
    if dr_names and rng.random() < cfg.dr_mutation_prob:
        current = child.genes[0].name.upper()
        candidates = [n for n in dr_names if n != current]
        if candidates:
            child.genes[0].name = rng.choice(candidates)

    # mutate MH genes
    for g in child.genes[1:]:
        g.w_raw = float(g.w_raw + rng.normal(loc=0.0, scale=cfg.mutation_sigma))
        g.w_raw = float(np.clip(g.w_raw, 0.1, 2.0))
        if mh_names and rng.random() < cfg.mh_name_mutation_prob:
            current = g.name.upper()
            candidates = [n for n in mh_names if n != current]
            if candidates:
                g.name = rng.choice(candidates)

    return child


def _crossover(parent_a: ActionIndividual,
               parent_b: ActionIndividual,
               rng: np.random.Generator,
               cfg: CoevolutionConfig) -> ActionIndividual:
    from lgp_actions import Gene
    genes = []

    # DR gene
    g0 = parent_a.genes[0] if rng.random() < 0.5 else parent_b.genes[0]
    genes.append(Gene(kind="DR", name=g0.name, w_raw=1.0))

    # MH genes
    for ga, gb in zip(parent_a.genes[1:], parent_b.genes[1:]):
        chosen = ga if rng.random() < 0.5 else gb
        name = chosen.name
        mean_w = 0.5 * (ga.w_raw + gb.w_raw)
        w = float(mean_w + rng.normal(loc=0.0, scale=cfg.mutation_sigma * 0.5))
        w = float(np.clip(w, 0.1, 2.0))
        genes.append(Gene(kind="MH", name=name, w_raw=w))

    child = ActionIndividual(genes=genes)
    child = _mutate_individual(child, rng, cfg)
    return child


def evolve_action_library(action_library: List[ActionIndividual],
                          fitness: np.ndarray,
                          usage: np.ndarray,
                          cfg: CoevolutionConfig,
                          rng: np.random.Generator) -> tuple[List[int], List[tuple]]:
    """
    Returns: (loser_indices, parent_pairs)
    """
    K = len(action_library)
    assert fitness.shape == (K,)
    assert usage.shape == (K,)

    elite_indices = _select_elite_indices(fitness, cfg.elite_size)

    r = fitness.copy()
    u = usage.astype(float).copy()

    if np.max(r) > np.min(r):
        r_norm = (r - np.min(r)) / (np.max(r) - np.min(r))
    else:
        r_norm = np.zeros_like(r)

    if np.max(u) > np.min(u):
        u_norm = (u - np.min(u)) / (np.max(u) - np.min(u))
    else:
        u_norm = np.zeros_like(u)

    w_r, w_u = 0.7, 0.3
    bad_score = w_r * (1.0 - r_norm) + w_u * (1.0 - u_norm)

    candidate_indices = [i for i in range(K) if i not in elite_indices]
    candidate_indices.sort(key=lambda i: bad_score[i], reverse=True)
    loser_indices = candidate_indices[:cfg.n_replace]

    parent_idx = _sample_parent_indices(fitness, rng, k=2 * cfg.n_replace)
    
    parent_pairs = []
    for j, loser in enumerate(loser_indices):
        pa_idx = parent_idx[2 * j]
        pb_idx = parent_idx[2 * j + 1]
        parent_pairs.append((pa_idx, pb_idx))
        
        pa = action_library[pa_idx]
        pb = action_library[pb_idx]
        child = _crossover(pa, pb, rng, cfg)
        action_library[loser] = child

    return loser_indices, parent_pairs


def train_with_coevolution(env,
                           action_library: List[ActionIndividual],
                           model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer,
                           select_action_fn: Callable,
                           compute_returns_fn: Callable,
                           cfg: CoevolutionConfig,
                           output_dir: str = "results/portfolios"):
    """
    env: DynamicSchedulingEnv (đã gắn action_library)
    select_action_fn: hàm select_action trong main.py
    compute_returns_fn: compute_returns trong main.py
    output_dir: thư mục lưu portfolios
    """
    K = len(action_library)
    rng = np.random.default_rng(seed=0)
    
    # ========== Save initial portfolios ==========
    print("\n" + "="*70)
    print("📦 INITIAL PORTFOLIOS (64 random portfolios)")
    print("="*70)
    
    initial_data = {
        "generation": 0,
        "description": "Initial random portfolios before training",
        "portfolios": [
            portfolio_to_dict(ind, index=i)
            for i, ind in enumerate(action_library)
        ]
    }
    
    initial_file = os.path.join(output_dir, "generation_0_initial.json")
    save_portfolios_json(initial_data, initial_file)
    
    # Print summary of initial portfolios
    print("\nSample initial portfolios:")
    for i in range(min(5, K)):
        print_portfolio(action_library[i], i, prefix="  ")

    for gen in range(cfg.num_generations):
        print(f"\n========== Generation {gen+1}/{cfg.num_generations} ==========")

        usage = np.zeros(K, dtype=np.int32)
        sum_reward = np.zeros(K, dtype=np.float32)
        episodes_metrics = []  # NEW: Collect metrics for each episode

        # --------- Train PPO trong 1 generation ----------
        for ep in range(cfg.episodes_per_gen):
            state = env.reset()

            states_list = []
            actions_list = []
            log_probs_list = []
            values_list = []
            rewards = []
            masks = []

            total_reward_ep = 0.0

            for step in range(cfg.max_steps_per_episode):
                action, log_prob, value = select_action_fn(model, state)

                next_state, reward, done, info = env.step(action)

                states_list.append(state)
                actions_list.append(action)
                log_probs_list.append(log_prob)
                values_list.append(value.squeeze(0))
                rewards.append(reward)
                masks.append(0.0 if done else 1.0)

                usage[action] += 1
                sum_reward[action] += reward

                total_reward_ep += reward
                state = next_state

                if done:
                    break

            returns = compute_returns_fn(rewards, masks, cfg.gamma)
            states_np = np.array(states_list, dtype=np.float32)
            states = torch.FloatTensor(states_np)
            actions_t = torch.LongTensor(actions_list)
            log_probs_old = torch.stack(log_probs_list).detach()
            values_old = torch.stack(values_list).detach().squeeze(-1)
            returns_t = torch.FloatTensor(returns)

            advantage = returns_t - values_old

            total_policy_loss = 0.0
            total_value_loss = 0.0
            count_updates = 0

            for _ in range(cfg.ppo_epochs):
                logits, value_est = model(states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                log_probs = dist.log_prob(actions_t)
                ratio = torch.exp(log_probs - log_probs_old)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio,
                                    1.0 - cfg.clip_epsilon,
                                    1.0 + cfg.clip_epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(value_est.squeeze(-1), returns_t)
                entropy = dist.entropy().mean()

                loss = policy_loss - cfg.entropy_coef * entropy + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                count_updates += 1

            avg_pl = total_policy_loss / max(count_updates, 1)
            avg_vl = total_value_loss / max(count_updates, 1)

            print(f"[Gen {gen+1} Ep {ep+1}] "
                  f"Return={total_reward_ep:.2f} | "
                  f"PolicyLoss={avg_pl:.4f} | ValueLoss={avg_vl:.4f}")
            
            # Collect episode metrics (including losses)
            ep_metrics = collect_episode_metrics(env)
            ep_metrics['policy_loss'] = round(float(avg_pl), 4)
            ep_metrics['value_loss'] = round(float(avg_vl), 4)
            ep_metrics['return'] = round(float(total_reward_ep), 2)
            episodes_metrics.append(ep_metrics)
        
        # Save generation metrics
        save_generation_metrics(gen+1, episodes_metrics, output_dir)

        # --------- Tính fitness & evolve ----------
        avg_reward = np.zeros(K, dtype=np.float32)
        for i in range(K):
            if usage[i] > 0:
                avg_reward[i] = sum_reward[i] / usage[i]
            else:
                avg_reward[i] = -1e9
        
        # Get elite indices before evolution
        elite_indices = _select_elite_indices(avg_reward, cfg.elite_size)
        
        # Save losers BEFORE they are replaced
        r_copy = avg_reward.copy()
        u_copy = usage.copy()
        
        # Calculate bad score to find losers
        r_norm = (r_copy - np.min(r_copy)) / (np.max(r_copy) - np.min(r_copy)) if np.max(r_copy) > np.min(r_copy) else np.zeros_like(r_copy)
        u_norm = (u_copy.astype(float) - np.min(u_copy)) / (np.max(u_copy) - np.min(u_copy)) if np.max(u_copy) > np.min(u_copy) else np.zeros_like(u_copy.astype(float))
        bad_score = 0.7 * (1.0 - r_norm) + 0.3 * (1.0 - u_norm)
        candidate_indices = [i for i in range(K) if i not in elite_indices]
        candidate_indices.sort(key=lambda i: bad_score[i], reverse=True)
        loser_indices_preview = candidate_indices[:cfg.n_replace]
        
        # Save loser portfolios before replacement
        losers_before = [
            portfolio_to_dict(action_library[idx], idx, avg_reward[idx], usage[idx])
            for idx in loser_indices_preview
        ]

        loser_indices, parent_pairs = evolve_action_library(
            action_library=action_library,
            fitness=avg_reward,
            usage=usage,
            cfg=cfg,
            rng=rng
        )
        
        # ========== Save generation summary ==========
        gen_data = {
            "generation": gen + 1,
            "elite_portfolios": [
                portfolio_to_dict(action_library[idx], idx, avg_reward[idx], usage[idx])
                for idx in elite_indices
            ],
            "loser_portfolios": losers_before,
            "new_portfolios": [
                {
                    **portfolio_to_dict(action_library[idx], idx),
                    "parents": list(parent_pairs[i])
                }
                for i, idx in enumerate(loser_indices)
            ]
        }
        
        gen_file = os.path.join(output_dir, f"generation_{gen+1}_summary.json")
        save_portfolios_json(gen_data, gen_file)
        
        # Print summary
        print_portfolios_summary(
            action_library=action_library,
            elite_indices=elite_indices,
            loser_indices=loser_indices,
            fitness=avg_reward,
            usage=usage,
            parent_pairs=parent_pairs
        )
        
        # Log top portfolios với normalized weights
        log_top_k_portfolios(
            action_library=action_library,
            fitnesses=avg_reward,
            k=5,  # Show top 5
            maximize=True  # reward càng lớn (ít âm hơn) càng tốt
        )

        print(f"\n[Gen {gen+1}] Evolve done. Replaced losers at indices: {loser_indices}")

        # --------- Warmup cho pool mới ----------
        for w in range(cfg.warmup_episodes):
            state = env.reset()
            for step in range(cfg.max_steps_per_episode):
                with torch.no_grad():
                    action, _, _ = select_action_fn(model, state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                if done:
                    break

    print("\n=== Coevolution Training Complete ===")
    
    # ========== Save final portfolios ==========
    print("\n" + "="*70)
    print("📦 FINAL PORTFOLIOS (after all generations)")
    print("="*70)
    
    # Calculate final fitness
    final_usage = np.zeros(K, dtype=np.int32)
    final_sum_reward = np.zeros(K, dtype=np.float32)
    
    # Run a few episodes to get final metrics
    for _ in range(5):
        state = env.reset()
        for step in range(cfg.max_steps_per_episode):
            with torch.no_grad():
                action, _, _ = select_action_fn(model, state)
            next_state, reward, done, info = env.step(action)
            final_usage[action] += 1
            final_sum_reward[action] += reward
            state = next_state
            if done:
                break
    
    final_fitness = np.zeros(K, dtype=np.float32)
    for i in range(K):
        if final_usage[i] > 0:
            final_fitness[i] = final_sum_reward[i] / final_usage[i]
        else:
            final_fitness[i] = -1e9
    
    final_data = {
        "generation": cfg.num_generations,
        "description": f"Final portfolios after {cfg.num_generations} generations",
        "portfolios": [
            portfolio_to_dict(ind, index=i, fitness=final_fitness[i], usage=int(final_usage[i]))
            for i, ind in enumerate(action_library)
        ]
    }
    
    portfolios_dir = os.path.join(output_dir, "portfolios")
    final_portfolio_file = os.path.join(portfolios_dir, f"generation_{cfg.num_generations}_final.json")
    save_portfolios_json(final_data, final_portfolio_file)
    
    # Print top portfolios
    final_elite = _select_elite_indices(final_fitness, min(10, K))
    print(f"\nTop 10 portfolios by final fitness:")
    for idx in final_elite:
        print_portfolio(action_library[idx], idx, final_fitness[idx], final_usage[idx], "  ")
    
    print("\n" + "="*70)
    
    # ========== Final Evaluation Episode ==========
    print("\n" + "="*70)
    print("🎯 FINAL EVALUATION - Running best policy on clean episode")
    print("="*70)
    
    # Run one final episode to get clean schedule
    state = env.reset()
    final_episode_reward = 0.0
    for step in range(cfg.max_steps_per_episode):
        with torch.no_grad():
            action, _, _ = select_action_fn(model, state)
        next_state, reward, done, info = env.step(action)
        final_episode_reward += reward
        state = next_state
        if done:
            break
    
    print(f"Final episode reward: {final_episode_reward:.2f}")
    
    # Save final results (schedule + metrics + Gantt)
    save_final_results(env, output_dir, cfg.num_generations)
    
    print("\n" + "="*70)


