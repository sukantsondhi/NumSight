# Runs the Number-Visualizer app end-to-end on Windows PowerShell
# - Ensures venv exists and dependencies are installed
# - Trains the model if missing
# - Starts the Flask server

$ErrorActionPreference = 'Stop'

function Ensure-Venv {
    if (Test-Path .venv) {
        return
    }
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    try {
        py -3.11 -m venv .venv
    } catch {
        Write-Warning "Python 3.11 not found, falling back to 3.10."
        py -3.10 -m venv .venv
    }
}

function Activate-Venv {
    & .\.venv\Scripts\Activate.ps1
}

function Ensure-Dependencies {
    Write-Host "Installing/validating dependencies..." -ForegroundColor Cyan
    python -m pip install --upgrade pip
    pip install -r requirements.txt
}

function Ensure-Model {
    $kerasModel = Join-Path 'models' 'mnist_model.keras'
    $h5Model = Join-Path 'models' 'mnist_model.h5'
    if (-not (Test-Path $kerasModel) -and -not (Test-Path $h5Model)) {
        Write-Host "Model not found; training now..." -ForegroundColor Yellow
        python train_model.py
    } else {
        Write-Host "Model file found; skipping training." -ForegroundColor Green
    }
}

# Main
Ensure-Venv
Activate-Venv
Ensure-Dependencies
Ensure-Model

Write-Host "Starting Flask app..." -ForegroundColor Cyan
python app.py
