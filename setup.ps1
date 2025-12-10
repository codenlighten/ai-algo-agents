# Installation and verification script for Windows

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "AI Research Agent System - Setup" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Check Python version
Write-Host ""
Write-Host "Checking Python version..." -ForegroundColor Yellow
python --version

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# Verify installations
Write-Host ""
Write-Host "Verifying installations..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
python -c "from rich import print; print('[green]Rich installed[/green]')"

# Check .env file
Write-Host ""
Write-Host "Checking .env configuration..." -ForegroundColor Yellow
if (Test-Path .env) {
    Write-Host "✓ .env file found" -ForegroundColor Green
    $envContent = Get-Content .env
    if ($envContent -match "OPENAI_API_KEY") {
        Write-Host "✓ OPENAI_API_KEY is set" -ForegroundColor Green
    } else {
        Write-Host "✗ OPENAI_API_KEY not found in .env" -ForegroundColor Red
        Write-Host "Please add your OpenAI API key to .env" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ .env file not found" -ForegroundColor Red
    Write-Host "Please create .env with OPENAI_API_KEY=your_key" -ForegroundColor Yellow
}

# Run tests
Write-Host ""
Write-Host "Running tests..." -ForegroundColor Yellow
pytest tests/test_system.py -v --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host "Some tests may fail without GPU" -ForegroundColor Yellow
}

# Success message
Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Ensure OPENAI_API_KEY is set in .env"
Write-Host "  2. Run: python main.py"
Write-Host "  3. Or try: python main.py --example"
Write-Host ""
