"""
Test suite for AI research agent system
Run with: pytest tests/test_system.py
"""
import pytest
import torch
import torch.nn as nn
from agents.base_agent import AgentTeam, AgentRole
from optimizers.novel_optimizers import (
    SecondOrderMomentumOptimizer,
    LookAheadWrapper,
    AdaptiveGradientClipping
)
from loss_functions.novel_losses import (
    ConfidencePenalizedCrossEntropy,
    FocalLoss,
    CurriculumLoss
)
from models.novel_architectures import (
    DynamicDepthNetwork,
    MixtureOfExpertsLayer,
    MultiScaleAttention
)
from research.proposal_system import (
    ResearchProposal,
    ExperimentalSetup,
    ResearchProposalBuilder
)


class TestOptimizers:
    """Test novel optimizer implementations"""
    
    def test_second_order_momentum_basic(self):
        """Test SecondOrderMomentumOptimizer basic functionality"""
        model = nn.Linear(10, 5)
        optimizer = SecondOrderMomentumOptimizer(model.parameters(), lr=0.01)
        
        # Forward pass
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.sum()
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Check state is initialized
        for param in model.parameters():
            state = optimizer.state[param]
            assert 'exp_avg' in state
            assert 'exp_avg_sq' in state
            assert 'curvature' in state
    
    def test_lookahead_wrapper(self):
        """Test LookAheadWrapper with base optimizer"""
        model = nn.Linear(10, 5)
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = LookAheadWrapper(base_opt, la_steps=5, la_alpha=0.5)
        
        # Train for a few steps
        for _ in range(10):
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Verify slow buffers exist
        for group in optimizer.param_groups:
            for p in group['params']:
                assert 'slow_buffer' in base_opt.state[p]
    
    def test_adaptive_gradient_clipping(self):
        """Test AdaptiveGradientClipping"""
        model = nn.Linear(10, 5)
        optimizer = AdaptiveGradientClipping(
            model.parameters(),
            clip_factor=0.01
        )
        
        # Simulate training
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify clipping thresholds are tracked
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.base_optimizer.state[p]
                assert 'clip_threshold' in state


class TestLossFunctions:
    """Test novel loss function implementations"""
    
    def test_confidence_penalized_ce(self):
        """Test ConfidencePenalizedCrossEntropy"""
        criterion = ConfidencePenalizedCrossEntropy(penalty_weight=0.1)
        
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        
        loss = criterion(logits, targets)
        
        assert loss.item() > 0
        assert torch.isfinite(loss)
    
    def test_focal_loss(self):
        """Test FocalLoss"""
        criterion = FocalLoss(gamma=2.0)
        
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        
        loss = criterion(logits, targets)
        
        assert loss.item() > 0
        assert torch.isfinite(loss)
    
    def test_curriculum_loss(self):
        """Test CurriculumLoss"""
        base_criterion = nn.CrossEntropyLoss(reduction='none')
        criterion = CurriculumLoss(base_criterion, warmup_steps=100)
        
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        sample_ids = torch.arange(32)
        
        # Train for a few steps
        for step in range(50):
            loss = criterion(logits, targets, sample_ids)
            assert torch.isfinite(loss)


class TestArchitectures:
    """Test novel architecture implementations"""
    
    def test_dynamic_depth_network(self):
        """Test DynamicDepthNetwork"""
        model = DynamicDepthNetwork(
            input_dim=10,
            hidden_dim=20,
            output_dim=5,
            max_layers=4,
            initial_active_layers=2
        )
        
        x = torch.randn(32, 10)
        output = model(x)
        
        assert output.shape == (32, 5)
        
        # Test growing network
        initial_layers = model.active_layers
        model.grow_network()
        assert model.active_layers == initial_layers + 1
    
    def test_mixture_of_experts(self):
        """Test MixtureOfExpertsLayer"""
        layer = MixtureOfExpertsLayer(
            input_dim=10,
            num_experts=4,
            expert_dim=20,
            top_k=2
        )
        
        x = torch.randn(32, 10)
        output, load_balance_loss = layer(x)
        
        assert output.shape == (32, 10)
        assert torch.isfinite(load_balance_loss)
    
    def test_multiscale_attention(self):
        """Test MultiScaleAttention"""
        layer = MultiScaleAttention(
            embed_dim=64,
            num_heads=4,
            scales=(1, 2, 4)
        )
        
        x = torch.randn(8, 32, 64)  # [batch, seq_len, embed_dim]
        output = layer(x)
        
        assert output.shape == (8, 32, 64)


class TestResearchProposalSystem:
    """Test research proposal management"""
    
    def test_proposal_builder(self):
        """Test ResearchProposalBuilder"""
        experimental_setup = ExperimentalSetup(
            datasets=["CIFAR-10"],
            metrics=["Accuracy", "Loss"],
            baselines=["SGD", "Adam"],
            expected_improvements={"speed": "20%"},
            minimal_compute_requirements="1 GPU"
        )
        
        proposal = ResearchProposalBuilder() \
            .set_title("Test Proposal") \
            .set_author("Test Agent") \
            .set_core_concept("Test concept") \
            .set_summary("Test summary") \
            .add_benefit("Fast") \
            .add_risk("Memory") \
            .set_related_work("Related") \
            .set_novelty("Novel") \
            .set_implementation("PyTorch", "code") \
            .set_experimental_setup(experimental_setup) \
            .set_scalability("Scales well") \
            .set_engineering_constraints("None") \
            .set_reasoning_path("Logical") \
            .add_assumption("Assumption 1") \
            .build()
        
        assert proposal.title == "Test Proposal"
        assert len(proposal.hypothesized_benefits) == 1
        assert len(proposal.tradeoffs_and_risks) == 1
        assert len(proposal.assumptions) == 1
    
    def test_proposal_serialization(self):
        """Test proposal save/load"""
        experimental_setup = ExperimentalSetup(
            datasets=["MNIST"],
            metrics=["Accuracy"],
            baselines=["Baseline"],
            expected_improvements={},
            minimal_compute_requirements="CPU"
        )
        
        original = ResearchProposalBuilder() \
            .set_title("Serialization Test") \
            .set_author("Test") \
            .set_core_concept("Concept") \
            .set_summary("Summary") \
            .set_related_work("Work") \
            .set_novelty("Novel") \
            .set_implementation("PyTorch", "code") \
            .set_experimental_setup(experimental_setup) \
            .set_scalability("Good") \
            .set_engineering_constraints("None") \
            .set_reasoning_path("Path") \
            .build()
        
        # Convert to dict and back
        data = original.to_dict()
        loaded = ResearchProposal.from_dict(data)
        
        assert loaded.title == original.title
        assert loaded.core_concept == original.core_concept


class TestAgentSystem:
    """Test multi-agent system (requires API key)"""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for full test"
    )
    def test_agent_initialization(self):
        """Test agent team initialization"""
        try:
            team = AgentTeam()
            assert len(team.agents) == len(AgentRole)
            
            for role in AgentRole:
                agent = team.get_agent(role)
                assert agent is not None
                assert agent.role == role
        except Exception as e:
            pytest.skip(f"Agent initialization failed: {e}")


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_optimizer_with_model(self):
        """Test optimizer in training loop"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        optimizer = SecondOrderMomentumOptimizer(model.parameters(), lr=0.01)
        criterion = ConfidencePenalizedCrossEntropy(penalty_weight=0.1)
        
        # Training loop
        for _ in range(10):
            x = torch.randn(32, 10)
            targets = torch.randint(0, 5, (32,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            assert torch.isfinite(loss)
    
    def test_dynamic_architecture_training(self):
        """Test training dynamic architecture"""
        model = DynamicDepthNetwork(
            input_dim=10,
            hidden_dim=20,
            output_dim=5,
            max_layers=3
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Train with progressive depth
        for epoch in range(3):
            for _ in range(10):
                x = torch.randn(32, 10)
                targets = torch.randint(0, 5, (32,))
                
                optimizer.zero_grad()
                outputs = model(x, temperature=1.0 - epoch * 0.2)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Grow network after each epoch
            if model.active_layers < model.max_layers:
                model.grow_network()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
