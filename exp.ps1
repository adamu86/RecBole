docker exec -it recbole-exp zsh -ic "python run_recbole.py"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Experiment finished successfully. Shutting down..."
    Stop-Computer -Force
}