# Quick Start Guide

## Installation

```bash
# Clone or navigate to the project
cd ai-algo-agents

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
# Create a .env file with:
# OPENAI_API_KEY=your_key_here
```

## Running the System

### Interactive Mode

```bash
python main.py
```

This launches an interactive session where you can:
1. Brainstorm research ideas with all agents
2. Generate complete research proposals
3. Query specific agents
4. Run example research scenarios

### Quick Example

```bash
python main.py --example
```

See a quick demonstration of the coordinated agent workflow.

### Generate Example Proposals

```bash
python examples/example_proposals.py
```

Creates and saves example research proposals for:
- Novel optimizers
- Novel loss functions

### Run Experiments

```bash
# Test novel optimizers
python examples/test_novel_optimizers.py

# Test novel loss functions
python examples/test_novel_losses.py
```

**Note:** Requires PyTorch and torchvision installed.

## Project Structure

```
ai-algo-agents/
├── agents/              # Agent implementations
│   ├── base_agent.py   # Core agent architecture
│   └── __init__.py
├── research/            # Research proposals
│   ├── proposal_system.py
│   ├── proposals/      # Saved proposals (JSON)
│   └── sessions/       # Research sessions
├── experiments/         # Experimental framework
│   ├── experiment_framework.py
│   └── results/        # Experiment results
├── optimizers/          # Novel optimizers
│   └── novel_optimizers.py
├── loss_functions/      # Novel loss functions
│   └── novel_losses.py
├── models/              # Novel architectures
│   └── novel_architectures.py
├── examples/            # Example usage
│   ├── example_proposals.py
│   ├── test_novel_optimizers.py
│   └── test_novel_losses.py
├── utils/               # Utilities
│   └── research_prompts.md
├── main.py             # Main entry point
├── requirements.txt
└── README.md
```

## Agent Roles

### 1. Python Engineering Agent
- Implements clean, production-ready code
- Focuses on correctness and testing
- Optimizes for GPU/TPU efficiency

### 2. AI Algorithms Agent
- Deep expertise in optimization theory
- Researches novel training methods
- Provides mathematical rigor

### 3. Systems Design Agent
- Distributed training strategies
- Scalability analysis
- Hardware utilization

### 4. Training Pipeline Agent
- End-to-end workflows
- Data curriculum strategies
- Training stability

### 5. Architecture Design Agent
- Novel model architectures
- Parameterization schemes
- Efficiency optimization

## Example Workflow

### 1. Brainstorm Ideas

```python
from agents.base_agent import AgentTeam

team = AgentTeam()
results = team.brainstorm(
    "Novel optimization methods for sparse models"
)

for agent, response in results.items():
    print(f"\n{agent}:\n{response}")
```

### 2. Generate Full Proposal

```python
proposal = team.research_proposal_workflow(
    "Second-order optimization with curvature adaptation"
)

# Access different perspectives
print(proposal['concept'])          # Algorithm agent's concept
print(proposal['implementation'])   # Python engineer's code
print(proposal['scalability'])      # Systems agent's analysis
print(proposal['experiments'])      # Training pipeline's validation plan
```

### 3. Query Specific Agent

```python
from agents.base_agent import AgentRole

algo_agent = team.get_agent(AgentRole.AI_ALGORITHMS)
response = algo_agent.query(
    "What are the theoretical convergence properties of adaptive optimizers?"
)
```

## Novel Implementations Included

### Optimizers
- **SecondOrderMomentumOptimizer**: Curvature-aware optimization
- **LookAheadWrapper**: Improved stability via slow/fast weights
- **AdaptiveGradientClipping**: Layer-wise gradient clipping
- **StochasticWeightAveraging**: Better generalization via weight averaging

### Loss Functions
- **ConfidencePenalizedCrossEntropy**: Improved calibration
- **FocalLoss**: Class imbalance handling
- **ContrastivePredictiveLoss**: Self-supervised learning
- **AdaptiveWingLoss**: Robust regression
- **NoiseContrastiveEstimation**: Efficient large-vocab training
- **CurriculumLoss**: Automated curriculum learning

### Architectures
- **DynamicDepthNetwork**: Adaptive network depth
- **MixtureOfExpertsLayer**: Sparse expert routing
- **AdaptiveComputationTime**: Input-dependent computation
- **HyperNetwork**: Dynamic weight generation
- **MultiScaleAttention**: Multi-resolution attention

## Tips

1. **Start Simple**: Use the interactive mode to explore
2. **Iterate**: Agents improve with follow-up questions
3. **Experiment**: Run the validation experiments to verify proposals
4. **Document**: Save important sessions and proposals
5. **Scale**: Test ideas on small datasets before scaling up

## Next Steps

- Add custom agents for your domain
- Implement proposed innovations
- Run large-scale experiments
- Contribute novel methods back to the library
- Explore multi-agent collaboration patterns
