# ML Models Data Reorganization

This document describes the reorganization of the data directory structure for the machine learning models used in the price prediction system.

## Overview

The data directory has been reorganized to use a zone-based folder structure instead of the previous flat structure with zone-specific filenames. This makes the codebase more maintainable and easier to extend with new zones.

## New Directory Structure

```
ml_models/
├── data/
│   ├── DK1/
│   │   ├── training_data.csv         # (renamed from DK1_19-23.csv)
│   │   ├── prediction_data.csv       # (renamed from DK1_24.csv)
│   │   └── prediction_data_normalized.csv  # (renamed from DK1_24-normalized.csv)
│   ├── DK2/
│   │   ├── training_data.csv
│   │   ├── prediction_data.csv
│   │   └── prediction_data_normalized.csv
│   ├── SE1/
│   │   └── ...
│   └── ...
├── informer/
│   ├── DK1/
│   │   ├── results/
│   │   │   └── checkpoint.pth        # (copied from informer/results/checkpoint.pth)
│   │   └── config.json              # (updated to point to DK1 folder)
│   └── ...
└── gru/
    ├── DK1/
    │   ├── results/
    │   │   └── gru_trained.pt        # (copied from gru/results/gru_trained.pt)
    │   └── ...
    └── ...
```

## Scripts

Several scripts were created to facilitate the reorganization:

1. `reorganize_data.py` - Creates the new folder structure and copies files to their zone-specific locations
2. `setup_model_weights.py` - Copies model checkpoints to zone-specific folders
3. `clean_up_data_files.py` - Removes the original files after verifying the new structure works

## Configuration Changes

All configuration files have been updated to use the new folder structure:

```json
// Before
{
  "root_path": "./data/",
  "data_path": "DK1_19-23.csv",
}

// After
{
  "root_path": "./data/DK1/",
  "data_path": "training_data.csv",
}
```

## Code Changes

The following files were updated to use the new folder structure:

1. `hybrid_model.py` - Updated file path references
2. `PredictionService.py` - Updated file path references

## Note on Model Weights

Currently, all zones are using copies of the same model weights. In the future, zone-specific models should be trained for better predictions.

## Verification

The new folder structure has been verified to work with the test mode of the hybrid model. Tests were run using:

```bash
python test_hybrid_model.py --zone DK1 --test-mode
```

## Next Steps

1. Train zone-specific models for more accurate predictions
2. Implement automatic model selection based on zone in the prediction service
