"""
Quick Proposal Generator - Focus on AQL v2.0
Generate ONE detailed proposal to start implementation immediately
"""
import os
from dotenv import load_dotenv
from agents.base_agent import AgentTeam
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import json
from pathlib import Path
from datetime import datetime

load_dotenv()
console = Console()

def generate_aql_v2_proposal():
    """Generate focused proposal for AQL v2.0 optimization"""
    
    console.print(Panel.fit(
        "[bold green]AQL v2.0 Proposal Generation[/bold green]\n"
        "[cyan]Optimizing Adaptive Query-Based Learning[/cyan]",
        border_style="green"
    ))
    
    # Initialize agents
    team = AgentTeam()
    
    query = """We successfully validated Adaptive Query-Based Learning (AQL) on MNIST:
- Achieved 94.00% accuracy vs 93.60% baseline (+0.40%)
- More stable training progression
- BUT: 10% computational overhead from Monte Carlo Dropout uncertainty estimation

Design AQL v2.0 with these requirements:
1. Reduce uncertainty estimation overhead to <5% (target: 3%)
2. Replace Monte Carlo Dropout with faster method (Laplace approximation, ensemble, or learned predictor)
3. Add curriculum learning component (easy → hard progression)
4. Meta-learn the data selection policy (when to query vs exploit)
5. Scale to WikiText-103 (100M tokens) and transformers
6. Must work with GPU batch processing
7. Target: 5x data efficiency (train on 20% of data, match 100% baseline)

Provide:
- Concrete implementation plan for efficient uncertainty estimation
- Mathematical/algorithmic details for fast approximations
- Integration strategy with curriculum learning
- GPU optimization techniques
- Validation experiment design on WikiText-103
- Code architecture for experiments/aql_v2/

Focus on PRACTICAL, IMPLEMENTABLE solutions that we can code this week."""
    
    console.print("\n[yellow]Consulting 5 specialist agents...[/yellow]\n")
    
    # Get agent responses
    results = team.brainstorm(query)
    
    # Save proposal
    output_dir = Path("research/proposals/efficiency_initiative")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    proposal_file = output_dir / "aql_v2_optimization_proposal.json"
    
    with open(proposal_file, 'w') as f:
        json.dump({
            'title': 'AQL v2.0: Efficient Uncertainty Estimation and Curriculum Integration',
            'query': query,
            'agent_responses': results,
            'generated_at': datetime.now().isoformat(),
            'priority': 'HIGH',
            'status': 'READY_FOR_IMPLEMENTATION'
        }, f, indent=2)
    
    # Also save markdown version for easy reading
    md_file = output_dir / "aql_v2_optimization_proposal.md"
    
    with open(md_file, 'w') as f:
        f.write(f"# AQL v2.0 Optimization Proposal\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Status:** Ready for Implementation\n\n")
        f.write(f"---\n\n")
        f.write(f"## Research Question\n\n{query}\n\n")
        f.write(f"---\n\n")
        
        for agent_role, response in results.items():
            f.write(f"## {agent_role.upper()}\n\n")
            f.write(f"{response}\n\n")
            f.write(f"---\n\n")
    
    # Display results
    console.print(f"\n[bold green]✅ PROPOSAL GENERATED[/bold green]\n")
    console.print(f"Saved to:\n  • JSON: {proposal_file}\n  • Markdown: {md_file}\n")
    
    # Display summaries
    for agent_role, response in results.items():
        # Show first 400 chars of each response
        preview = response[:400] + "..." if len(response) > 400 else response
        console.print(Panel(
            Markdown(preview),
            title=f"🤖 {agent_role}",
            border_style="blue"
        ))
    
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print("[bold green]NEXT STEPS:[/bold green]")
    console.print("  1. Review proposal in research/proposals/efficiency_initiative/")
    console.print("  2. Implement efficient uncertainty estimation")
    console.print("  3. Set up WikiText-103 experiments")
    console.print("  4. Begin coding experiments/aql_v2/")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")
    
    return proposal_file

if __name__ == "__main__":
    try:
        generate_aql_v2_proposal()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Generation interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise
