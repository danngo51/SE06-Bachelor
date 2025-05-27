#!/bin/pwsh
# Script to test the pipeline functionality

Write-Host "=== Testing ML Pipeline Framework ===" -ForegroundColor Green

# 1. Train models for DK1
Write-Host "`nTraining models for DK1..." -ForegroundColor Cyan
python train_all_models.py DK1

# 2. Test hybrid model with all pipelines
Write-Host "`nTesting hybrid model with all pipelines..." -ForegroundColor Cyan
python test_hybrid_model_all_pipelines.py --country DK1 --date 2025-03-01

# 3. Test hybrid model with different weightings
Write-Host "`nTesting hybrid model with different weightings..." -ForegroundColor Cyan
python test_hybrid_model_all_pipelines.py --country DK1 --date 2025-03-01 --xgboost-weight 0.7 --gru-weight 0.15 --informer-weight 0.15
python test_hybrid_model_all_pipelines.py --country DK1 --date 2025-03-01 --xgboost-weight 0.2 --gru-weight 0.4 --informer-weight 0.4

# 4. Test prediction service
Write-Host "`nTesting prediction service..." -ForegroundColor Cyan
python test_hybrid_model.py

Write-Host "`nAll tests completed!" -ForegroundColor Green
