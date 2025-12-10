"""
Enhanced Research Proposal Generator
Supports both automated generation AND using local documents as inspiration
"""
import os
import argparse
from dotenv import load_dotenv
from agents.base_agent import AgentTeam
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import json
from pathlib import Path

load_dotenv()
console = Console()


def load_document_as_inspiration(doc_path: str) -> dict:
    """
    Load a local markdown document and extract key concepts
    
    Args:
        doc_path: Path to markdown file
        
    Returns:
        Dict with title, core concepts, and requirements
    """
    doc_path = Path(doc_path)
    
    if not doc_path.exists():
        console.print(f"[red]Error: Document not found at {doc_path}[/red]")
        return None
    
    console.print(f"\n[cyan]📄 Loading document: {doc_path.name}[/cyan]")
    
    with open(doc_path, 'r') as f:
        content = f.read()
    
    # Extract title (first heading)
    lines = content.split('\n')
    title = "Unknown"
    for line in lines:
        if line.startswith('# '):
            title = line[2:].strip()
            break
    
    console.print(f"[green]✓ Loaded: {title}[/green]")
    console.print(f"[dim]Content: {len(content)} characters, {len(lines)} lines[/dim]")
    
    return {
        "title": title,
        "content": content,
        "path": str(doc_path),
        "lines": len(lines),
        "chars": len(content)
    }


def generate_from_document(doc_info: dict, team: AgentTeam, output_dir: Path) -> Path:
    """
    Generate a research proposal inspired by a local document
    
    Args:
        doc_info: Document information from load_document_as_inspiration()
        team: AgentTeam instance
        output_dir: Directory to save proposals
        
    Returns:
        Path to generated proposal file
    """
    console.print("\n" + "=" * 70)
    console.print(f"[bold cyan]Generating Proposal from Document[/bold cyan]")
    console.print(f"[cyan]Source: {doc_info['title']}[/cyan]")
    console.print("=" * 70)
    
    # Create query for agents
    query = f"""Analyze and refine the following research concept into a comprehensive, implementable proposal:

DOCUMENT TITLE: {doc_info['title']}

DOCUMENT CONTENT:
{doc_info['content'][:5000]}  # First 5000 chars to avoid token limits

YOUR TASK:
1. Extract the core innovation and key ideas
2. Identify technical challenges and requirements
3. Design a practical implementation approach
4. Define clear experiments and success metrics
5. Assess feasibility and risks
6. Propose concrete next steps

Focus on making this implementable in our AI research lab with:
- PyTorch framework
- GPU acceleration (NVIDIA RTX 3070, 8GB VRAM)
- WikiText-103 dataset (101M tokens available)
- Target: <5% computational overhead, measurable efficiency gains

Provide a complete, production-ready research proposal."""
    
    console.print("\n[yellow]🤔 Agents analyzing document...[/yellow]\n")
    
    # Brainstorm with all agents
    results = team.brainstorm(query)
    
    # Display results
    for agent_name, response in results.items():
        panel = Panel(
            Markdown(response[:500] + "..." if len(response) > 500 else response),
            title=f"🤖 {agent_name}",
            border_style="cyan"
        )
        console.print(panel)
    
    # Create proposal structure
    from datetime import datetime
    proposal = {
        "source_document": {
            "title": doc_info['title'],
            "path": doc_info['path'],
            "size": f"{doc_info['chars']} chars, {doc_info['lines']} lines"
        },
        "generated_date": datetime.now().isoformat(),
        "agent_responses": results,
        "synthesis": {
            "title": doc_info['title'],
            "core_innovation": results.get('ai_algorithms', '')[:500],
            "implementation": results.get('python_engineer', '')[:500],
            "architecture": results.get('architecture_design', '')[:500],
            "training_pipeline": results.get('training_pipeline', '')[:500],
            "system_design": results.get('systems_design', '')[:500]
        }
    }
    
    # Save proposal
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from title
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in doc_info['title'])
    safe_title = safe_title.replace(' ', '_').lower()[:100]
    
    output_file = output_dir / f"proposal_from_{safe_title}.json"
    
    with open(output_file, 'w') as f:
        json.dump(proposal, f, indent=2)
    
    console.print(f"\n[green]✅ Saved to: {output_file}[/green]")
    
    return output_file


def generate_automated_proposals(team: AgentTeam, output_dir: Path, topics: list) -> list:
    """
    Generate proposals from predefined topics (original functionality)
    
    Args:
        team: AgentTeam instance
        output_dir: Directory to save proposals
        topics: List of research topics
        
    Returns:
        List of generated proposal files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    console.print(Panel.fit(
        "[bold cyan]AI Research Lab - Efficiency Initiative[/bold cyan]\n"
        "Automated Research Proposal Generation\n\n"
        f"Generating {len(topics)} comprehensive proposals with multi-agent collaboration",
        border_style="cyan"
    ))
    
    for i, topic in enumerate(topics, 1):
        console.print(f"\n[bold]Progress: {i}/{len(topics)}[/bold]\n")
        console.print("=" * 70)
        console.print(f"[cyan]Generating Proposal {i}: {topic['title']}[/cyan]")
        console.print("=" * 70)
        
        # Brainstorm
        console.print("\n[yellow]🤔 Agents collaborating...[/yellow]\n")
        results = team.brainstorm(topic['query'])
        
        # Display summaries
        for agent_name, response in results.items():
            panel = Panel(
                Markdown(response[:500] + "..." if len(response) > 500 else response),
                title=f"🤖 {agent_name}",
                border_style="cyan"
            )
            console.print(panel)
        
        # Save proposal
        proposal = {
            "id": topic['id'],
            "title": topic['title'],
            "query": topic['query'],
            "agent_responses": results
        }
        
        safe_title = topic['title'].replace(' ', '_').lower()
        output_file = output_dir / f"proposal_{topic['id']}_{safe_title}.json"
        
        with open(output_file, 'w') as f:
            json.dump(proposal, f, indent=2)
        
        console.print(f"\n[green]✅ Saved to: {output_file}[/green]")
        generated_files.append(output_file)
        
        console.print("\n[dim]Preparing next proposal...[/dim]\n")
    
    return generated_files


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Generate research proposals using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate proposals from predefined topics
  python generate_proposals_enhanced.py --mode auto
  
  # Generate proposal from a local document
  python generate_proposals_enhanced.py --mode document --doc hierarchical_contextual_encoding_*.md
  
  # Generate multiple proposals from documents
  python generate_proposals_enhanced.py --mode document --doc doc1.md --doc doc2.md
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['auto', 'document', 'both'],
        default='auto',
        help='Generation mode: auto (predefined topics), document (from files), or both'
    )
    
    parser.add_argument(
        '--doc',
        action='append',
        help='Path to markdown document(s) to use as inspiration (can specify multiple)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='research/proposals/generated',
        help='Output directory for proposals (default: research/proposals/generated)'
    )
    
    args = parser.parse_args()
    
    # Initialize agent team
    console.print("\n[cyan]🤖 Initializing AI Agent Team...[/cyan]")
    team = AgentTeam()
    console.print("[green]✓ Team ready![/green]")
    
    output_dir = Path(args.output_dir)
    
    # Mode: Document
    if args.mode == 'document' or args.mode == 'both':
        if not args.doc:
            console.print("[red]Error: --doc required for document mode[/red]")
            return
        
        for doc_path in args.doc:
            doc_info = load_document_as_inspiration(doc_path)
            if doc_info:
                generate_from_document(doc_info, team, output_dir)
    
    # Mode: Auto
    if args.mode == 'auto' or args.mode == 'both':
        # Predefined topics (from original script)
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
            }
        ]
        
        generate_automated_proposals(team, output_dir, topics)
    
    console.print("\n" + "=" * 70)
    console.print("[bold green]✅ All proposals generated successfully![/bold green]")
    console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
    console.print("=" * 70)


if __name__ == "__main__":
    main()
