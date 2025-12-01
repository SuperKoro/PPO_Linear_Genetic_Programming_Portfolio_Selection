# PPO + LGP Dynamic Job Shop Scheduling

> A hybrid reinforcement learning and evolutionary approach combining Proximal Policy Optimization (PPO) with Linear Genetic Programming (LGP) for dynamic job shop scheduling problems.

## 🎯 Overview

This project tackles the **Dynamic Job Shop Scheduling Problem** where:
- 20 initial jobs need to be scheduled across multiple machines
- 2-4 new jobs arrive dynamically during execution
- The system must reschedule unfinished jobs when disruptions occur
- **Goal**: Minimize makespan (total completion time)

### Key Innovation

Instead of learning low-level actions, the PPO agent learns to select **high-level portfolios** where each portfolio is a combination of:
- **1 Dispatching Rule (DR)** - for job ordering (e.g., EDD, SPT, CR)
- **3 Metaheuristics (MH)** - for optimization (e.g., SA, GA, PSO) with weighted time budgets

Example portfolio: `EDD | SA:52%, GA:25%, PSO:23%`

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           Coevolution Training Loop              │
│                                                  │
│  Generation 1 → 2 → ... → 10                    │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │ PPO Agent                                   │ │
│  │ • Observes state (time, #jobs, avg_pt)     │ │
│  │ • Selects 1 of 64 portfolios               │ │
│  │ • Learns which portfolio for which state   │ │
│  └────────────────────────────────────────────┘ │
│                       ↕                          │
│  ┌────────────────────────────────────────────┐ │
│  │ LGP Evolution                               │ │
│  │ • Maintains 64 portfolios                   │ │
│  │ • Keeps top 16 elite (best fitness)        │ │
│  │ • Replaces 4 worst via crossover/mutation  │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Result**: PPO learns *when* to use which portfolio, while LGP evolves *better* portfolios over time.

## 📁 Project Structure

### Core Files

| File | Description |
|------|-------------|
| `main.py` | Entry point - initializes PPO + LGP and runs training |
| `lgp_actions.py` | Defines Gene, ActionIndividual (portfolio), ActionLGP |
| `coevolution_trainer.py` | Training loop with coevolution logic |
| `typed_action_adapter.py` | Executes portfolios (DR + MH pipeline) |

### Registry System Files

| File | Description |
|------|-------------|
| `dispatching_registry.py` | Registry framework for dispatching rules |
| `dispatching_rules.py` | Implements and registers DR functions (EDD, SPT, CR, etc.) |
| `mh_registry.py` | Registry framework for metaheuristics |
| `metaheuristics_impl.py` | Implements and registers MH functions (SA, GA, PSO, etc.) |

### Utility Files

| File | Description |
|------|-------------|
| `visualize_metrics.py` | Plot training metrics across generations |
| `inference.py` | Run trained model for testing |
| `sa_scheduling.py` | Standalone SA implementation for comparison |
| `excel_to_json.py` | Convert Excel data to JSON format |

## 🔧 Registry Pattern Explained

The project uses the **Registry Pattern** to manage dispatching rules and metaheuristics dynamically.

### How It Works

**1. Registry Framework** (`dispatching_registry.py`)
```python
# Global registry to store functions
DR_REGISTRY: Dict[str, Callable] = {}

def register_dr(name: str):
    """Decorator to auto-register dispatching rules"""
    def deco(fn):
        DR_REGISTRY[name.upper()] = fn
        return fn
    return deco

def get_dr(name: str):
    """Retrieve function by name"""
    return DR_REGISTRY[name.upper()]
```

**2. Function Registration** (`dispatching_rules.py`)
```python
@register_dr("EDD")  # ← Auto-registers when imported
def dr_edd(env, finished_events, unfinished_jobs, time_budget_s=0.0):
    """Earliest Due Date dispatching rule"""
    return reschedule_unfinished_jobs_edd(...)

@register_dr("SPT")
def dr_spt(env, finished_events, unfinished_jobs, time_budget_s=0.0):
    """Shortest Processing Time dispatching rule"""
    return reschedule_unfinished_jobs_spt(...)
```

**3. Runtime Lookup** (`typed_action_adapter.py`)
```python
# When executing a portfolio
dr_name = portfolio.dr_gene.name  # e.g., "EDD"
dr_fn = get_dr(dr_name)           # ← Dynamic lookup
result = dr_fn(env, ...)          # ← Execute

# Same for metaheuristics
mh_name = gene.name               # e.g., "SA"
mh_fn = get_mh(mh_name)          # ← Dynamic lookup
result = mh_fn(env, ...)         # ← Execute
```

### Why Registry Pattern?

✅ **Decoupling**: Separate function definition from usage  
✅ **Extensibility**: Add new DR/MH without modifying existing code  
✅ **Dynamic Selection**: LGP can evolve portfolios with any registered DR/MH  
✅ **Clean Code**: No hard-coded if-else chains

**Adding a new rule is simple:**
```python
@register_dr("FIFO")  # Just add decorator - done!
def dr_fifo(env, finished_events, unfinished_jobs, time_budget_s=0.0):
    # Implementation...
    pass
```

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/SuperKoro/PPO_Linear_Genetic_Programming_Portfolio_Selection.git
cd PPO_Linear_Genetic_Programming_Portfolio_Selection

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- numpy >= 1.21.0
- torch >= 1.10.0
- gym >= 0.21.0
- matplotlib >= 3.5.0
- openpyxl >= 3.0.0

## 📊 Usage

### Training
```bash
python main.py
```

**Training Parameters** (in `main.py`):
- `num_generations = 10`: Number of coevolution cycles
- `episodes_per_gen = 100`: Episodes per generation
- `pool_size = 64`: Number of portfolios
- `elite_size = 16`: Top portfolios to keep
- `n_replace = 4`: Portfolios to replace each generation

**Output**:
- `results/generation_X_summary.json`: Metrics for each generation
- `results/portfolios/generation_X_final.json`: Final portfolio states
- `trained_policy_coevolution.pth`: Trained PPO model

### Visualization
```bash
python visualize_metrics.py
```

Displays:
- Average reward over generations
- Policy loss and value loss
- Makespan trends
- Portfolio fitness evolution

### Inference
```bash
python inference.py
```

Loads trained model and runs test episodes with visualization.

## 📈 Components Deep Dive

### Portfolio Structure

Each portfolio (ActionIndividual) consists of:
```python
Portfolio {
    DR Gene: (kind="DR", name="EDD", weight=1.0)
    MH Gene 1: (kind="MH", name="SA", weight=0.52)
    MH Gene 2: (kind="MH", name="GA", weight=0.25)
    MH Gene 3: (kind="MH", name="PSO", weight=0.23)
}
```

### Execution Pipeline

When a portfolio is executed:
1. **Stage 0 (DR)**: Dispatching rule sorts jobs by priority
2. **Stage 1 (MH1)**: First metaheuristic optimizes with 52% of time budget
3. **Stage 2 (MH2)**: Second metaheuristic refines with 25% of budget
4. **Stage 3 (MH3)**: Third metaheuristic finalizes with 23% of budget

### PPO State & Action

**State (Observation)**:
```python
[current_time, num_unfinished_operations, avg_processing_time]
```

**Action Space**: Discrete(64) - choose one of 64 portfolios

This project is licensed under the MIT License - see the LICENSE file for details.
