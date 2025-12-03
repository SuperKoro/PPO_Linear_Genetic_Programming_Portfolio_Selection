# LGP Coevolution Training Module

from typing import List, Dict, Any
import os
import json
import random
import numpy as np
import torch

from config import LGPConfig
from lgp_program import LGPProgram
from lgp_generator import LGPGenerator
from lgp_evolution import linear_crossover, mutate_program
from lgp_actions import ActionIndividual, describe_individual
from coevolution_trainer import (
    CoevolutionConfig,
    _select_elite_indices,
    collect_episode_metrics,
    save_generation_metrics,
    portfolio_to_dict,
    save_portfolios_json,
    save_final_results
)


def build_lgp_inputs_for_env(env) -> Dict[str, float]:
    """
    Xây macro-state cho LGP.
    Có thể chỉnh lại tuỳ dạng dữ liệu jobs của bạn.
    """
    num_jobs = 0
    total_pt = 0.0
    total_ops = 0

    jobs = getattr(env, "jobs_initial", None)
    if isinstance(jobs, dict):
        num_jobs = len(jobs)
        for job_id, job_info in jobs.items():
            ops = job_info.get("operations") if isinstance(job_info, dict) else job_info
            if ops is None:
                continue
            for op in ops:
                pt = op.get("processing_time", 0.0) if isinstance(op, dict) else 0.0
                total_pt += float(pt)
                total_ops += 1
    elif isinstance(jobs, list):
        num_jobs = len(jobs)
        for job in jobs:
            ops = job.get("operations") if isinstance(job, dict) else None
            if ops is None:
                continue
            for op in ops:
                pt = op.get("processing_time", 0.0) if isinstance(op, dict) else 0.0
                total_pt += float(pt)
                total_ops += 1

    avg_pt = total_pt / total_ops if total_ops > 0 else 0.0
    avg_ops_per_job = total_ops / num_jobs if num_jobs > 0 else 0.0

    return {
        "num_jobs": float(num_jobs),
        "avg_processing_time": float(avg_pt),
        "avg_ops_per_job": float(avg_ops_per_job),
    }


def make_fallback_individual() -> ActionIndividual:
    """
    Portfolio fallback nếu LGPProgram bị lỗi runtime.
    EDD + 1 SA, các MH còn lại weight 0.
    """
    from lgp_actions import Gene
    genes = [
        Gene(kind="DR", name=LGPConfig.available_dr[0], w_raw=1.0),
        Gene(kind="MH", name=LGPConfig.available_mh[0], w_raw=1.0),
    ]
    while len(genes) < 1 + LGPConfig.n_mh_genes:
        genes.append(Gene(kind="MH", name=LGPConfig.available_mh[0], w_raw=0.0))
    return ActionIndividual(genes=genes)


def train_with_coevolution_lgp(
    env,
    lgp_programs: List[LGPProgram],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    select_action_fn,
    compute_returns_fn,
    cfg: CoevolutionConfig,
    output_dir: str = "results_lgp",
):
    """
    Coevolution training: evolve LGP programs, PPO vẫn chọn index action như cũ.
    """
    os.makedirs(output_dir, exist_ok=True)

    K = len(lgp_programs)
    assert K == LGPConfig.pool_size, "Pool size must match number of LGP programs"

    rng_np = np.random.default_rng(seed=0)
    rng_py = random.Random(0)

    lgp_gen = LGPGenerator(
        max_length=LGPConfig.max_program_length,
        min_length=LGPConfig.min_program_length,
        num_registers=LGPConfig.num_registers,
        rng=rng_py,
    )

    for gen in range(cfg.num_generations):
        print(f"\n========== LGP Generation {gen+1}/{cfg.num_generations} ==========")

        # 1) Sinh portfolios từ programs
        lgp_inputs = build_lgp_inputs_for_env(env)

        action_library: List[ActionIndividual] = []
        for idx, prog in enumerate(lgp_programs):
            try:
                ind = prog.execute(lgp_inputs)
            except Exception as e:
                print(f"[WARN] Program {idx} crashed during execute(): {e}")
                ind = make_fallback_individual()
            action_library.append(ind)

        # gắn vào env
        env.action_library = action_library

        usage = np.zeros(K, dtype=np.int32)
        sum_reward = np.zeros(K, dtype=np.float32)
        episodes_metrics = []

        # 2) PPO training trong generation này
        for ep in range(cfg.episodes_per_gen):
            state = env.reset()

            states_list = []
            actions_list = []
            log_probs_list = []
            values_list = []
            rewards = []
            masks = []

            ep_return = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            count_updates = 0

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
                ep_return += reward

                state = next_state
                if done:
                    break

            # PPO update cho episode này
            returns = compute_returns_fn(rewards, masks, gamma=cfg.gamma)
            returns_t = torch.tensor(returns, dtype=torch.float32)
            values_t = torch.stack(values_list)
            log_probs_old = torch.stack(log_probs_list).detach()  # DETACH to avoid graph issues
            # Convert to numpy first to avoid slow tensor creation
            states_np = np.array(states_list, dtype=np.float32)
            states_t = torch.from_numpy(states_np)
            actions_t = torch.tensor(actions_list, dtype=torch.int64)

            advantages = returns_t - values_t.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            value_coef = 0.5
            for _ in range(cfg.ppo_epochs):
                logits, values = model(states_t)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions_t)
                ratio = torch.exp(log_probs - log_probs_old)

                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - cfg.clip_epsilon,
                    1.0 + cfg.clip_epsilon,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (returns_t - values.squeeze(-1)).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + value_coef * value_loss - cfg.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                count_updates += 1
            
            avg_pl = total_policy_loss / max(count_updates, 1)
            avg_vl = total_value_loss / max(count_updates, 1)
            
            # Collect episode metrics
            ep_metrics = collect_episode_metrics(env)
            ep_metrics["policy_loss"] = avg_pl
            ep_metrics["value_loss"] = avg_vl
            ep_metrics["return"] = ep_return  # ADD episode return for metrics tracking
            episodes_metrics.append(ep_metrics)
            
            if (ep + 1) % 50 == 0:
                print(f"[Gen {gen+1} Ep {ep+1}] Return={ep_return:.2f} | PolicyLoss={avg_pl:.4f} | ValueLoss={avg_vl:.4f}")

        # 3) Fitness cho mỗi program
        avg_reward = np.full(K, -1e9, dtype=np.float32)
        for i in range(K):
            if usage[i] > 0:
                avg_reward[i] = sum_reward[i] / max(1, usage[i])

        # 4) Evolve LGP programs
        elite_indices = _select_elite_indices(avg_reward, cfg.elite_size)

        best_idx = int(elite_indices[0])
        print(f"\n[Gen {gen+1}] Best program idx={best_idx}, avg_reward={avg_reward[best_idx]:.3f}")
        print("  Example portfolio from best program:")
        print("   ", describe_individual(action_library[best_idx]))

        # Save generation metrics
        save_generation_metrics(gen + 1, episodes_metrics, output_dir)
        
        # Save portfolios
        portfolios_data = {
            "generation": gen + 1,
            "elite": [portfolio_to_dict(action_library[i], i, avg_reward[i], usage[i]) for i in elite_indices],
            "all_fitness": avg_reward.tolist(),
            "all_usage": usage.tolist()
        }
        portfolios_dir = os.path.join(output_dir, "portfolios")
        os.makedirs(portfolios_dir, exist_ok=True)
        save_portfolios_json(portfolios_data, os.path.join(portfolios_dir, f"generation_{gen+1}_final.json"))

        # ===================================================================
        # DIVERSITY MECHANISM 1: Protect unused programs in early generations
        # ===================================================================
        def _should_protect_program(prog_idx, gen_num):
            """Protect programs with low usage to prevent PPO bias from eliminating unexplored programs"""
            prog_usage = usage[prog_idx]
            
            if gen_num <= 3:
                # First 3 generations: protect completely unused
                return prog_usage == 0
            elif gen_num <= 6:
                # Generations 4-6: protect rarely used (< 3 times)
                return prog_usage < 3
            else:
                # After gen 6: normal selection pressure
                return False
        
        # Select candidates for replacement (non-elite programs)
        candidate_indices = [i for i in range(K) if i not in elite_indices]
        
        # Separate into protected and unprotected
        protected_programs = [i for i in candidate_indices if _should_protect_program(i, gen + 1)]
        unprotected_programs = [i for i in candidate_indices if not _should_protect_program(i, gen + 1)]
        
        # Select losers only from unprotected pool
        if len(unprotected_programs) >= cfg.n_replace:
            # Enough unprotected programs - use them
            unprotected_programs.sort(key=lambda i: avg_reward[i])  # Sort by fitness (worst first)
            loser_indices = unprotected_programs[:cfg.n_replace]
        else:
            # Not enough unprotected - use all candidates (protection overridden)
            candidate_indices.sort(key=lambda i: avg_reward[i])
            loser_indices = candidate_indices[:cfg.n_replace]
            if protected_programs:
                print(f"  [WARNING] Gen {gen+1}: Had to replace {len([i for i in loser_indices if i in protected_programs])} protected programs")

        # ===================================================================
        # DIVERSITY MECHANISM 2: Rank-based parent selection
        # ===================================================================
        def _sample_parents_rank_based(num_pairs: int):
            """
            Sample parents using rank-based selection instead of fitness-proportional.
            This reduces selection pressure and gives lower-fitness programs more chance.
            """
            # Sort programs by fitness (descending order - best first)
            sorted_indices = np.argsort(avg_reward)[::-1]
            
            # Assign ranks: best program = rank K, worst = rank 1
            ranks = np.arange(len(sorted_indices), 0, -1)
            
            # Selection probability proportional to rank (not exponential like softmax)
            # This is MUCH less biased than fitness-proportional
            rank_probs = ranks / ranks.sum()
            
            pairs = []
            for _ in range(num_pairs):
                # Sample two parents based on rank probabilities
                try:
                    selected = rng_np.choice(sorted_indices, size=2, p=rank_probs, replace=False)
                    p1, p2 = int(selected[0]), int(selected[1])
                except ValueError:
                    # Fallback if something goes wrong
                    p1 = int(sorted_indices[0])
                    p2 = int(sorted_indices[min(1, len(sorted_indices)-1)])
                
                pairs.append((p1, p2))
            
            return pairs

        # Sample parent pairs using rank-based selection
        parent_pairs = _sample_parents_rank_based(cfg.n_replace)
        
        # ===================================================================
        # DIVERSITY MECHANISM 3: Limit best program copies
        # ===================================================================
        best_program_idx = int(elite_indices[0])
        max_copies_from_best = K // 4  # Maximum 25% of population from same parent
        
        # Count how many children would come from best program
        best_program_usage_in_pairs = sum(
            1 for p1, p2 in parent_pairs 
            if p1 == best_program_idx or p2 == best_program_idx
        )
        
        # If too many pairs use best program, redistribute some
        if best_program_usage_in_pairs > max_copies_from_best:
            print(f"  [DIVERSITY] Gen {gen+1}: Limiting best program #{best_program_idx} from {best_program_usage_in_pairs} to {max_copies_from_best} children")
            
            # Get other top-performing programs to use instead
            other_good_programs = [int(i) for i in elite_indices[1:6] if i != best_program_idx]  # Top 2-6
            
            if len(other_good_programs) >= 2:
                new_pairs = []
                best_count = 0
                
                for p1, p2 in parent_pairs:
                    uses_best = (p1 == best_program_idx or p2 == best_program_idx)
                    
                    if uses_best and best_count >= max_copies_from_best:
                        # Replace this pair with pair from other good programs
                        new_p1, new_p2 = rng_np.choice(other_good_programs, size=2, replace=False)
                        new_pairs.append((int(new_p1), int(new_p2)))
                    else:
                        new_pairs.append((p1, p2))
                        if uses_best:
                            best_count += 1
                
                parent_pairs = new_pairs

        new_programs = list(lgp_programs)
        for li, (p1, p2) in zip(loser_indices, parent_pairs):
            child = linear_crossover(lgp_programs[p1], lgp_programs[p2], rng_py)
            child = mutate_program(
                child,
                generator=lgp_gen,
                rng=rng_py,
                mutation_rate=LGPConfig.mutation_rate,
            )
            new_programs[li] = child

        lgp_programs = new_programs

    # Save final results
    save_final_results(env, output_dir, cfg.num_generations)
    
    return lgp_programs, action_library
