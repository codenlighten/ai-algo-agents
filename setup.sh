#!/bin/bash
# Installation and verification script for AI Research Agent System

echo "=================================="
echo "AI Research Agent System - Setup"
echo "=================================="

# Check Python version
echo ""
echo "Checking Python version..."
python --version

# Create virtual environment (optional but recommended)
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/Scripts/activate || source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
python -c "from rich import print; print('[green]Rich installed[/green]')"

# Check .env file
echo ""
echo "Checking .env configuration..."
if [ -f .env ]; then
    echo "✓ .env file found"
    if grep -q "OPENAI_API_KEY" .env; then
        echo "✓ OPENAI_API_KEY is set"
    else
        echo "✗ OPENAI_API_KEY not found in .env"
        echo "Please add your OpenAI API key to .env"
    fi
else
    echo "✗ .env file not found"
    echo "Please create .env with OPENAI_API_KEY=your_key"
fi

# Run tests
echo ""
echo "Running tests..."
pytest tests/test_system.py -v --tb=short || echo "Some tests may fail without GPU"

# Success message
echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Ensure OPENAI_API_KEY is set in .env"
echo "  2. Run: python main.py"
echo "  3. Or try: python main.py --example"
echo ""
