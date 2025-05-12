# Data Folder Structure

This folder contains zone-specific data organized in subfolders:

## Structure
Each zone has its own subfolder (e.g., DK1, SE1) containing:
- `training_data.csv` - Historical data from 2019-2023 used for training models
- `prediction_data.csv` - Recent data from 2024 used for predictions
- `prediction_data_normalized.csv` - Normalized version of the prediction data (if available)

## Usage
The config files have been updated to point to these new locations.
