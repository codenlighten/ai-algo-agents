"""
Main orchestration script for AI research agent team
"""
import os
from dotenv import load_dotenv
from agents.base_agent import AgentTeam, AgentRole
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress
import json

# Load environment variables
load_dotenv()

console = Console()


def display_agent_response(role: str, response: str):
    """Display agent response with rich formatting"""
    console.print(Panel(
        Markdown(response),
        title=f"🤖 {role}",
        border_style="blue"
    ))


def save_research_session(topic: str, results: dict):
    """Save research session results"""
    os.makedirs("research/sessions", exist_ok=True)
    filename = f"research/sessions/{topic.replace(' ', '_')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n💾 Session saved to: {filename}", style="green")


def interactive_research_session():
    """Interactive research session with agent team"""
    console.print(Panel.fit(
        "[bold blue]AI Algorithm Research Agent Team[/bold blue]\n"
        "Exploring innovations beyond standard training paradigms",
        border_style="blue"
    ))
    
    # Initialize agent team
    console.print("\n🚀 Initializing research agent team...")
    team = AgentTeam()
    
    console.print("\n✅ Team assembled with specialists in:")
    for role in AgentRole:
        console.print(f"  • {role.value.replace('_', ' ').title()}")
    
    while True:
        console.print("\n" + "="*80)
        console.print("\n[bold]Research Options:[/bold]")
        console.print("1. Brainstorm new training innovation")
        console.print("2. Full research proposal workflow")
        console.print("3. Query specific agent")
        console.print("4. Example: Novel optimizer research")
        console.print("5. Example: Novel loss function research")
        console.print("6. Example: Novel architecture research")
        console.print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "7":
            console.print("\n👋 Research session ended.", style="yellow")
            break
        
        elif choice == "1":
            topic = input("\n📝 Enter research topic: ").strip()
            console.print(f"\n🧠 Team brainstorming on: {topic}\n")
            
            results = team.brainstorm(topic)
            
            for role, response in results.items():
                display_agent_response(role, response)
            
            save = input("\n💾 Save this session? (y/n): ").strip().lower()
            if save == 'y':
                save_research_session(topic, results)
        
        elif choice == "2":
            topic = input("\n📝 Enter research topic for full proposal: ").strip()
            console.print(f"\n🔬 Generating complete research proposal for: {topic}\n")
            
            results = team.research_proposal_workflow(topic)
            
            console.print("\n" + "="*80)
            console.print("[bold green]COMPLETE RESEARCH PROPOSAL[/bold green]")
            console.print("="*80)
            
            for section, content in results.items():
                console.print(f"\n[bold cyan]{section.upper()}[/bold cyan]")
                console.print(content)
                console.print("\n" + "-"*80)
            
            save = input("\n💾 Save this proposal? (y/n): ").strip().lower()
            if save == 'y':
                save_research_session(topic, results)
        
        elif choice == "3":
            console.print("\n[bold]Available agents:[/bold]")
            for i, role in enumerate(AgentRole, 1):
                console.print(f"{i}. {role.value.replace('_', ' ').title()}")
            
            agent_choice = input("\nSelect agent (1-5): ").strip()
            roles = list(AgentRole)
            
            try:
                selected_role = roles[int(agent_choice) - 1]
                question = input("\n📝 Your question: ").strip()
                
                agent = team.get_agent(selected_role)
                console.print(f"\n🤔 {selected_role.value} is thinking...\n")
                response = agent.query(question)
                
                display_agent_response(selected_role.value, response)
            except (IndexError, ValueError):
                console.print("\n❌ Invalid agent selection", style="red")
        
        elif choice == "4":
            console.print("\n🔬 Researching novel optimization methods...\n")
            topic = "alternative optimization algorithms beyond Adam/SGD that could improve convergence speed and stability"
            results = team.research_proposal_workflow(topic)
            
            for section, content in results.items():
                display_agent_response(section, content)
            
            save_research_session("novel_optimizer", results)
        
        elif choice == "5":
            console.print("\n🔬 Researching novel loss functions...\n")
            topic = "novel loss functions that improve generalization and robustness"
            results = team.research_proposal_workflow(topic)
            
            for section, content in results.items():
                display_agent_response(section, content)
            
            save_research_session("novel_loss", results)
        
        elif choice == "6":
            console.print("\n🔬 Researching novel architectures...\n")
            topic = "novel neural network architectures with better efficiency and expressiveness"
            results = team.research_proposal_workflow(topic)
            
            for section, content in results.items():
                display_agent_response(section, content)
            
            save_research_session("novel_architecture", results)
        
        else:
            console.print("\n❌ Invalid option", style="red")


def quick_example():
    """Quick example demonstrating the system"""
    console.print(Panel.fit(
        "[bold blue]Quick Example: Novel Optimizer Research[/bold blue]",
        border_style="blue"
    ))
    
    team = AgentTeam()
    
    # AI Algorithms agent proposes
    console.print("\n[bold]Step 1: AI Algorithms Agent - Core Concept[/bold]")
    algo_agent = team.get_agent(AgentRole.AI_ALGORITHMS)
    concept = algo_agent.query(
        "Propose a novel second-order optimization method that approximates "
        "curvature information efficiently without computing full Hessian"
    )
    console.print(Panel(Markdown(concept), border_style="green"))
    
    # Python engineer implements
    console.print("\n[bold]Step 2: Python Engineer - Implementation[/bold]")
    py_agent = team.get_agent(AgentRole.PYTHON_ENGINEER)
    implementation = py_agent.query(
        f"Implement this optimizer in PyTorch:\n{concept}"
    )
    console.print(Panel(Markdown(implementation), border_style="blue"))
    
    console.print("\n✅ Example complete! See the coordinated workflow.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        quick_example()
    else:
        interactive_research_session()
