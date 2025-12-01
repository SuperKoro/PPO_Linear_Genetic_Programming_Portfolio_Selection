# config.py
"""
Configuration file for PPO + LGP Dynamic Job Shop Scheduling

⚠️ WARNING: Some parameters are critical for system stability!
Read comments carefully before modifying.
"""

# ============================================================================
# 🎯 EXPERIMENT SETTINGS (Safe to modify)
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Output directory for results
OUTPUT_DIR = "results"

# Model save path
MODEL_SAVE_PATH = "trained_policy_coevolution.pth"


# ============================================================================
# 🧠 PPO HYPERPARAMETERS
# ============================================================================

class PPOConfig:
    """PPO Agent hyperparameters"""
    
    # Learning rate (Safe to tune: 1e-5 to 1e-3)
    learning_rate = 3e-4
    
    # Discount factor (⚠️ CAREFUL: affects long-term planning)
    # Recommended range: 0.85 - 0.99
    gamma = 0.9
    
    # Number of PPO update epochs per generation
    # Safe to tune: 3-15
    ppo_epochs = 10
    
    # PPO clip parameter (⚠️ CRITICAL: too high = unstable training)
    # ⛔ DO NOT CHANGE unless you know what you're doing!
    # Recommended: 0.1 - 0.3
    clip_epsilon = 0.25
    
    # Entropy coefficient for exploration
    # Safe to tune: 0.001 - 0.05
    entropy_coef = 0.01
    
    # Generalized Advantage Estimation lambda (⚠️ CAREFUL)
    # Recommended: 0.9 - 0.99
    gae_lambda = 0.95


# ============================================================================
# 🧬 COEVOLUTION PARAMETERS
# ============================================================================

class CoevolutionConfig:
    """Coevolution hyperparameters"""
    
    # Number of generations (Safe to modify)
    # Training time scales linearly with this
    num_generations = 10
    
    # Episodes per generation (Safe to modify)
    # More episodes = better fitness estimate but slower
    episodes_per_gen = 200
    
    # Maximum steps per episode (⚠️ CAREFUL)
    # Too low: episodes end prematurely
    # Too high: wasted computation
    max_steps_per_episode = 200
    
    # ⛔ CRITICAL PARAMETERS - DO NOT CHANGE WITHOUT GOOD REASON ⛔
    
    # Number of elite portfolios to keep (⚠️ CRITICAL)
    # MUST be < pool_size and > n_replace
    # Recommended: 20-30% of pool_size
    elite_size = 16
    
    # Number of portfolios to replace each generation (⚠️ CRITICAL)
    # MUST be < elite_size
    # Recommended: 5-10% of pool_size
    n_replace = 4
    
    # Warmup episodes before starting evolution
    # Safe to modify: 1-5
    warmup_episodes = 2
    
    # Mutation strength (Safe to tune: 0.1 - 0.5)
    mutation_sigma = 0.3
    
    # Probability to mutate dispatching rule (Safe to tune: 0.05 - 0.2)
    dr_mutation_prob = 0.1
    
    # Probability to mutate MH name (Safe to tune: 0.1 - 0.3)
    mh_name_mutation_prob = 0.2


# ============================================================================
# 📦 LGP ACTION LIBRARY CONFIGURATION
# ============================================================================

class LGPConfig:
    """LGP and Portfolio configuration"""
    
    # ⛔ CRITICAL: Total number of portfolios ⛔
    # MUST be even number and >= elite_size + n_replace
    # Recommended: 32, 64, 128
    # ⚠️ Larger = more diversity but slower training
    pool_size = 64
    
    # Number of MH genes per portfolio (⚠️ CAREFUL)
    # ⛔ Changing this requires code changes in typed_action_adapter.py!
    # DO NOT MODIFY unless you update execution pipeline
    n_mh_genes = 3
    
    # Available Dispatching Rules (Safe to modify)
    # Add/remove rules as needed
    available_dr = ["EDD", "SPT", "LPT", "FCFS", "CR"]
    
    # Available Metaheuristics (Safe to modify)
    # Add/remove MHs as needed (must be registered first!)
    available_mh = ["SA", "GA", "PSO", "EDD"]
    
    # Time budget for portfolio execution (⚠️ CAREFUL)
    # Too low: MHs can't optimize properly
    # Too high: slow training
    # Recommended: 0.5 - 3.0 seconds
    action_budget_s = 3.0


# ============================================================================
# 🏭 ENVIRONMENT PARAMETERS
# ============================================================================

class EnvironmentConfig:
    """Job Shop Scheduling Environment settings"""
    
    # Lambda for tardiness penalty (Safe to tune: 0.5 - 2.0)
    lambda_tardiness = 1.0
    
    # Number of dynamic jobs per episode (⚠️ CAREFUL)
    # Affects episode difficulty
    # Recommended: 2-4
    num_dynamic_jobs = 2
    
    # Observation space size (⚠️ CRITICAL - DO NOT CHANGE)
    # ⛔ Changing this requires model architecture changes!
    obs_dim = 3  # [current_time, num_unfinished, avg_processing_time]


# ============================================================================
# ⚠️ VALIDATION FUNCTIONS
# ============================================================================

def validate_config():
    """
    Validate configuration parameters for consistency.
    Raises ValueError if configuration is invalid.
    """
    errors = []
    
    # Check pool_size constraints
    if LGPConfig.pool_size < CoevolutionConfig.elite_size + CoevolutionConfig.n_replace:
        errors.append(
            f"⛔ CRITICAL: pool_size ({LGPConfig.pool_size}) must be >= "
            f"elite_size ({CoevolutionConfig.elite_size}) + "
            f"n_replace ({CoevolutionConfig.n_replace})"
        )
    
    if LGPConfig.pool_size % 2 != 0:
        errors.append(f"⚠️ WARNING: pool_size ({LGPConfig.pool_size}) should be even number")
    
    # Check elite_size constraints
    if CoevolutionConfig.elite_size <= CoevolutionConfig.n_replace:
        errors.append(
            f"⛔ CRITICAL: elite_size ({CoevolutionConfig.elite_size}) must be > "
            f"n_replace ({CoevolutionConfig.n_replace})"
        )
    
    # Check PPO clip epsilon
    if PPOConfig.clip_epsilon > 0.5 or PPOConfig.clip_epsilon < 0.05:
        errors.append(
            f"⚠️ WARNING: clip_epsilon ({PPOConfig.clip_epsilon}) outside "
            f"recommended range [0.05, 0.5]"
        )
    
    # Check gamma
    if PPOConfig.gamma < 0.8 or PPOConfig.gamma > 0.99:
        errors.append(
            f"⚠️ WARNING: gamma ({PPOConfig.gamma}) outside "
            f"recommended range [0.8, 0.99]"
        )
    
    # Check action budget
    if LGPConfig.action_budget_s < 0.1:
        errors.append(
            f"⚠️ WARNING: action_budget_s ({LGPConfig.action_budget_s}s) is very low. "
            f"Metaheuristics may not have enough time to optimize."
        )
    
    if errors:
        print("\n" + "=" * 70)
        print("⚠️  CONFIGURATION WARNINGS/ERRORS ⚠️")
        print("=" * 70)
        for error in errors:
            print(f"  {error}")
        print("=" * 70 + "\n")
        
        # Raise error only for CRITICAL issues
        critical_errors = [e for e in errors if "CRITICAL" in e]
        if critical_errors:
            raise ValueError("Configuration has critical errors. Please fix them!")
    else:
        print("✅ Configuration validated successfully!")


def print_config_summary():
    """Print configuration summary for verification"""
    print("\n" + "=" * 70)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print("\n🧠 PPO Configuration:")
    print(f"  Learning Rate:     {PPOConfig.learning_rate}")
    print(f"  Gamma:             {PPOConfig.gamma}")
    print(f"  PPO Epochs:        {PPOConfig.ppo_epochs}")
    print(f"  Clip Epsilon:      {PPOConfig.clip_epsilon}")
    print(f"  Entropy Coef:      {PPOConfig.entropy_coef}")
    
    print("\n🧬 Coevolution Configuration:")
    print(f"  Generations:       {CoevolutionConfig.num_generations}")
    print(f"  Episodes/Gen:      {CoevolutionConfig.episodes_per_gen}")
    print(f"  Elite Size:        {CoevolutionConfig.elite_size}")
    print(f"  Replace Count:     {CoevolutionConfig.n_replace}")
    print(f"  Mutation Sigma:    {CoevolutionConfig.mutation_sigma}")
    
    print("\n📦 LGP Configuration:")
    print(f"  Pool Size:         {LGPConfig.pool_size}")
    print(f"  MH Genes:          {LGPConfig.n_mh_genes}")
    print(f"  Action Budget:     {LGPConfig.action_budget_s}s")
    print(f"  Available DR:      {', '.join(LGPConfig.available_dr)}")
    print(f"  Available MH:      {', '.join(LGPConfig.available_mh)}")
    
    print("\n🏭 Environment Configuration:")
    print(f"  Dynamic Jobs:      {EnvironmentConfig.num_dynamic_jobs}")
    print(f"  Lambda Tardiness:  {EnvironmentConfig.lambda_tardiness}")
    
    print("\n🎯 Experiment Settings:")
    print(f"  Random Seed:       {RANDOM_SEED}")
    print(f"  Output Dir:        {OUTPUT_DIR}")
    
    total_episodes = CoevolutionConfig.num_generations * CoevolutionConfig.episodes_per_gen
    print(f"\n📊 Total Training Episodes: {total_episodes}")
    print("=" * 70 + "\n")


# ============================================================================
# 🔧 QUICK PRESETS
# ============================================================================

class Presets:
    """Pre-configured settings for different use cases"""
    
    @staticmethod
    def quick_test():
        """Fast training for testing (low quality)"""
        CoevolutionConfig.num_generations = 3
        CoevolutionConfig.episodes_per_gen = 20
        LGPConfig.pool_size = 32
        print("✅ Applied QUICK TEST preset")
    
    @staticmethod
    def standard():
        """Standard training (good balance)"""
        CoevolutionConfig.num_generations = 10
        CoevolutionConfig.episodes_per_gen = 100
        LGPConfig.pool_size = 64
        print("✅ Applied STANDARD preset")
    
    @staticmethod
    def high_quality():
        """High quality training (slow but better results)"""
        CoevolutionConfig.num_generations = 20
        CoevolutionConfig.episodes_per_gen = 200
        LGPConfig.pool_size = 128
        CoevolutionConfig.elite_size = 32
        CoevolutionConfig.n_replace = 8
        print("✅ Applied HIGH QUALITY preset")


# ============================================================================
# 🚀 AUTO-RUN ON IMPORT
# ============================================================================

if __name__ == "__main__":
    # When run directly, validate and print config
    print_config_summary()
    validate_config()
else:
    # When imported, only validate
    validate_config()
