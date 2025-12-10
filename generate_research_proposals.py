"""
Automated Research Proposal Generation
Generate 5 key proposals for our efficiency research initiative
"""
import os
from dotenv import load_dotenv
from agents.base_agent import AgentTeam
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import json
from pathlib import Path

load_dotenv()
console = Console()

# Initialize agent team
team = AgentTeam()

# Research topics for our initiative
topics = [
    {
        "id": 1,
        "title": "Efficient Uncertainty Estimation for LLM Active Learning",
        "query": """Design efficient uncertainty estimation methods for active learning in large language model training.
        
        Requirements:
        - Must have <5% computational overhead (current MC Dropout is 10%)
        - Should work with transformer architectures
        - Needs to scale to billion-parameter models
        - Must identify truly informative training samples
        
        Consider: Laplace approximation, ensemble methods, single-pass uncertainty, learned uncertainty predictors.
        Focus on practical implementation for GPU acceleration."""
    },
    {
        "id": 2,
        "title": "Meta-Learning Optimizers for Transformer Training",
        "query": """Design a meta-learning system that learns how to optimize transformer model training.
        
        Requirements:
        - Learn optimization strategy from small models, apply to large ones
        - Should outperform Adam/AdamW in convergence speed
        - Must be memory efficient
        - Should adapt per-layer and per-parameter
        
        Consider: LSTM-based optimizers, reinforcement learning for learning rate schedules, learned momentum strategies.
        Target: 20-30% faster convergence than Adam."""
    },
    {
        "id": 3,
        "title": "Dynamic Architecture Growth Policies",
        "query": """Design policies for growing transformer architectures during training based on information bottlenecks.
        
        Requirements:
        - Start with small model (4 layers), grow to target size (12-24 layers)
        - Detect when to add capacity (layers, heads, hidden dims)
        - Efficient knowledge transfer during growth
        - Must reduce total training FLOPs by 30-40%
        
        Consider: Reinforcement learning for growth decisions, gradient flow analysis, attention pattern analysis.
        Focus on practical triggers for when to grow."""
    },
    {
        "id": 4,
        "title": "Adaptive Curriculum Learning for Language Models",
        "query": """Design curriculum learning system that adaptively orders training data by difficulty for LLMs.
        
        Requirements:
        - Automatically assess data difficulty without manual labeling
        - Adapt curriculum based on model's learning progress
        - Integrate with active learning (AQL)
        - Should accelerate early-stage training 2-3x
        
        Consider: Loss-based difficulty, perplexity gradients, uncertainty measures, meta-learned difficulty predictors.
        Must work with streaming data (can't sort entire dataset upfront)."""
    },
    {
        "id": 5,
        "title": "Integrated Efficient Training System Architecture",
        "query": """Design system that integrates adaptive architecture, smart data selection, meta-learned optimizers, and curriculum learning.
        
        Requirements:
        - Components must work together synergistically
        - Unified API for easy experimentation
        - Must achieve 10x efficiency improvement over baseline
        - Should be compatible with Hugging Face ecosystem
        
        Consider: Coordination mechanisms, conflict resolution (e.g., data selection vs curriculum), shared state management.
        Focus on achieving emergent benefits from integration (whole > sum of parts)."""
    }
]

def generate_proposal(topic):
    """Generate research proposal using agent team"""
    console.print(f"\n[bold blue]{'='*70}[/bold blue]")
    console.print(f"[bold cyan]Generating Proposal {topic['id']}: {topic['title']}[/bold cyan]")
    console.print(f"[bold blue]{'='*70}[/bold blue]\n")
    
    # Get multi-agent brainstorming
    results = team.brainstorm(topic['query'])
    
    # Save proposal
    output_dir = Path("research/proposals/efficiency_initiative")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    proposal_file = output_dir / f"proposal_{topic['id']}_{topic['title'].lower().replace(' ', '_')}.json"
    
    from datetime import datetime
    
    with open(proposal_file, 'w') as f:
        json.dump({
            'id': topic['id'],
            'title': topic['title'],
            'query': topic['query'],
            'agent_responses': results,
            'generated_at': datetime.now().isoformat()
        }, f, indent=2)
    
    console.print(f"\n[green]✅ Saved to: {proposal_file}[/green]\n")
    
    # Display summary
    for agent_role, response in results.items():
        console.print(Panel(
            Markdown(response[:500] + "..." if len(response) > 500 else response),
            title=f"🤖 {agent_role}",
            border_style="blue"
        ))
    
    return proposal_file

def main():
    console.print(Panel.fit(
        "[bold green]AI Research Lab - Efficiency Initiative[/bold green]\n"
        "[cyan]Automated Research Proposal Generation[/cyan]\n\n"
        "Generating 5 comprehensive proposals with multi-agent collaboration",
        border_style="green"
    ))
    
    generated_files = []
    
    for i, topic in enumerate(topics, 1):
        console.print(f"\n[bold yellow]Progress: {i}/{len(topics)}[/bold yellow]")
        
        try:
            proposal_file = generate_proposal(topic)
            generated_files.append(proposal_file)
            
            # Brief pause between proposals
            console.print("\n[dim]Preparing next proposal...[/dim]")
            
        except Exception as e:
            console.print(f"[bold red]Error generating proposal {i}: {e}[/bold red]")
            continue
    
    # Summary
    console.print(f"\n[bold green]{'='*70}[/bold green]")
    console.print(f"[bold green]GENERATION COMPLETE[/bold green]")
    console.print(f"[bold green]{'='*70}[/bold green]\n")
    
    console.print(f"Generated {len(generated_files)} proposals:")
    for i, file in enumerate(generated_files, 1):
        console.print(f"  {i}. {file}")
    
    console.print(f"\n[cyan]Next steps:[/cyan]")
    console.print("  1. Review proposals in research/proposals/efficiency_initiative/")
    console.print("  2. Select most promising approaches")
    console.print("  3. Begin implementation planning")
    console.print("  4. Set up experiments\n")

if __name__ == "__main__":
    main()
