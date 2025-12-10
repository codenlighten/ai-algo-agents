"""
Research proposal templates and management system
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os


@dataclass
class ExperimentalSetup:
    """Experimental validation setup"""
    datasets: List[str]
    metrics: List[str]
    baselines: List[str]
    expected_improvements: Dict[str, str]
    minimal_compute_requirements: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ResearchProposal:
    """Comprehensive research proposal structure"""
    
    # Identification
    proposal_id: str
    title: str
    created_at: str
    author_agent: str
    
    # Core concept
    core_concept: str
    high_level_summary: str
    
    # Benefits and trade-offs
    hypothesized_benefits: List[str]
    tradeoffs_and_risks: List[str]
    
    # Literature and novelty
    related_work: str
    novelty_statement: str
    
    # Technical implementation
    implementation_language: str
    code_snippet: str
    
    # Experimental validation
    experimental_setup: ExperimentalSetup
    
    # Scalability and engineering
    scalability_analysis: str
    engineering_constraints: str
    
    # Reasoning and assumptions
    reasoning_path: str
    assumptions: List[str]
    
    # Optional fields
    safety_considerations: Optional[str] = None
    alignment_implications: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['experimental_setup'] = self.experimental_setup.to_dict()
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, directory: str = "research/proposals"):
        """Save proposal to file"""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.proposal_id}.json")
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        return filepath
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResearchProposal':
        """Create proposal from dictionary"""
        # Make a copy to avoid modifying the original
        data_copy = data.copy()
        exp_setup_data = data_copy.pop('experimental_setup')
        
        # Handle if exp_setup_data is already an ExperimentalSetup instance
        if isinstance(exp_setup_data, ExperimentalSetup):
            exp_setup = exp_setup_data
        else:
            exp_setup = ExperimentalSetup(**exp_setup_data)
        
        return cls(**data_copy, experimental_setup=exp_setup)
    
    @classmethod
    def load(cls, filepath: str) -> 'ResearchProposal':
        """Load proposal from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ResearchProposalBuilder:
    """Builder pattern for creating research proposals"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._proposal_data = {
            'proposal_id': f"proposal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'created_at': datetime.now().isoformat(),
            'hypothesized_benefits': [],
            'tradeoffs_and_risks': [],
            'assumptions': []
        }
    
    def set_title(self, title: str) -> 'ResearchProposalBuilder':
        self._proposal_data['title'] = title
        return self
    
    def set_author(self, author_agent: str) -> 'ResearchProposalBuilder':
        self._proposal_data['author_agent'] = author_agent
        return self
    
    def set_core_concept(self, concept: str) -> 'ResearchProposalBuilder':
        self._proposal_data['core_concept'] = concept
        return self
    
    def set_summary(self, summary: str) -> 'ResearchProposalBuilder':
        self._proposal_data['high_level_summary'] = summary
        return self
    
    def add_benefit(self, benefit: str) -> 'ResearchProposalBuilder':
        self._proposal_data['hypothesized_benefits'].append(benefit)
        return self
    
    def add_risk(self, risk: str) -> 'ResearchProposalBuilder':
        self._proposal_data['tradeoffs_and_risks'].append(risk)
        return self
    
    def set_related_work(self, related_work: str) -> 'ResearchProposalBuilder':
        self._proposal_data['related_work'] = related_work
        return self
    
    def set_novelty(self, novelty: str) -> 'ResearchProposalBuilder':
        self._proposal_data['novelty_statement'] = novelty
        return self
    
    def set_implementation(self, language: str, code: str) -> 'ResearchProposalBuilder':
        self._proposal_data['implementation_language'] = language
        self._proposal_data['code_snippet'] = code
        return self
    
    def set_experimental_setup(self, setup: ExperimentalSetup) -> 'ResearchProposalBuilder':
        self._proposal_data['experimental_setup'] = setup
        return self
    
    def set_scalability(self, analysis: str) -> 'ResearchProposalBuilder':
        self._proposal_data['scalability_analysis'] = analysis
        return self
    
    def set_engineering_constraints(self, constraints: str) -> 'ResearchProposalBuilder':
        self._proposal_data['engineering_constraints'] = constraints
        return self
    
    def set_reasoning_path(self, reasoning: str) -> 'ResearchProposalBuilder':
        self._proposal_data['reasoning_path'] = reasoning
        return self
    
    def add_assumption(self, assumption: str) -> 'ResearchProposalBuilder':
        self._proposal_data['assumptions'].append(assumption)
        return self
    
    def set_safety_considerations(self, safety: str) -> 'ResearchProposalBuilder':
        self._proposal_data['safety_considerations'] = safety
        return self
    
    def build(self) -> ResearchProposal:
        """Build the final proposal"""
        proposal = ResearchProposal.from_dict(self._proposal_data)
        self.reset()
        return proposal


class ProposalLibrary:
    """Manage collection of research proposals"""
    
    def __init__(self, directory: str = "research/proposals"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
    
    def add_proposal(self, proposal: ResearchProposal) -> str:
        """Add proposal to library"""
        return proposal.save(self.directory)
    
    def get_proposal(self, proposal_id: str) -> Optional[ResearchProposal]:
        """Get proposal by ID"""
        filepath = os.path.join(self.directory, f"{proposal_id}.json")
        if os.path.exists(filepath):
            return ResearchProposal.load(filepath)
        return None
    
    def list_proposals(self) -> List[ResearchProposal]:
        """List all proposals"""
        proposals = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.json'):
                filepath = os.path.join(self.directory, filename)
                proposals.append(ResearchProposal.load(filepath))
        return proposals
    
    def search_by_keyword(self, keyword: str) -> List[ResearchProposal]:
        """Search proposals by keyword"""
        results = []
        for proposal in self.list_proposals():
            if (keyword.lower() in proposal.title.lower() or
                keyword.lower() in proposal.core_concept.lower()):
                results.append(proposal)
        return results
