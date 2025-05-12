# Zone-Specific Model Structure

This document explains the new zone-specific structure for the ML models in the project.

## Overview

The project has been restructured to:

1. Use non-normalized data (removing normalization layer)
2. Support zone-specific models and data files
3. Train and predict for each zone independently
4. Use zone-specific model weights and configurations

## Directory Structure

The new directory structure is as follows:

```
ml_models/
  ├── data/
  │   ├── DK1_19-23.csv       # Training data for DK1 (2019-2023)
  │   ├── DK1_24.csv          # Prediction/validation data for DK1 (2024)
  │   ├── DK2_19-23.csv       # Training data for DK2 (2019-2023)
  │   ├── DK2_24.csv          # Prediction/validation data for DK2 (2024)
  │   └── ...                 # Other zone files
  │
  ├── informer/
  │   ├── config.json         # Default config file (fallback)
  │   ├── results/            # Default results folder (fallback)
  │   ├── DK1/               
  │   │   ├── config.json     # Zone-specific config for DK1
  │   │   └── results/        # Zone-specific model weights for DK1
  │   ├── DK2/
  │   │   ├── config.json
  │   │   └── results/
  │   └── ...                 # Other zones
  │
  ├── gru/
  │   ├── results/            # Default results folder (fallback)
  │   ├── DK1/
  │   │   └── results/        # Zone-specific GRU model for DK1
  │   ├── DK2/
  │   │   └── results/
  │   └── ...                 # Other zones
  │
  ├── hybrid_model.py         # Updated hybrid model that supports zones
  ├── setup_zones.py          # Script to set up zone structure
  ├── train_zone_models.py    # Script to train zone-specific models
  └── test_hybrid_model.py    # Script to test zone-specific predictions
```

## Migration Guide

Follow these steps to migrate your existing models to the new structure:

1. Run `setup_zones.py` to create the zone-specific structure
2. Run `convert_data_to_zones.py` to convert existing normalized data to zone-specific files
3. For each zone, train a separate model using `train_zone_models.py`
4. Test your models using `test_hybrid_model.py`

## Using Zone-Specific Models

The hybrid_model.py file now automatically:

1. Looks for zone-specific data files
2. Attempts to load zone-specific models
3. Falls back to default models if zone-specific ones aren't available
4. Uses zone-specific weights for combining model outputs

## Zone-Specific Weight Configuration

Each zone can have different weights for combining the Informer and GRU model outputs. These weights are defined in `hybrid_model.py`.

## Testing

Use the `test_hybrid_model.py` script to test predictions for specific zones:

```bash
python test_hybrid_model.py --zone DK1 --test-mode  # Use test mode
python test_hybrid_model.py --zone DK1  # Use real model
```

## Training

Use the `train_zone_models.py` script to train models for specific zones:

```bash
python train_zone_models.py --zone DK1 --model informer --epochs 20  # Train Informer
python train_zone_models.py --zone DK1 --model gru --epochs 20       # Train GRU
python train_zone_models.py --zone DK1 --model both --epochs 20      # Train both
```
