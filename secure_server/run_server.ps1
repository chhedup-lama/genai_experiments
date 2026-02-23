param(
    [int]$Port = 8000
)

if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Installing server dependencies..."
pip install --upgrade pip | Out-Null
pip install -r "secure_server\requirements.txt"

if (-not $env:SECURE_SERVER_API_KEY) {
    $env:SECURE_SERVER_API_KEY = [guid]::NewGuid().ToString()
    Write-Host "Generated SECURE_SERVER_API_KEY (save this value):"
    Write-Host $env:SECURE_SERVER_API_KEY
}

Write-Host "Starting secure FastAPI server on port $Port..."
uvicorn secure_server.app:app --host 0.0.0.0 --port $Port

