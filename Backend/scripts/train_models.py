import os
import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml_models.XGBoost.XGBoost_model import XGBoostRegimeModel
from ml_models.GRU.GRU_model import GRUModelTrainer
from ml_models.Informer.Informer_model import InformerModelTrainer

def train_xgboost(country_code):
    model = XGBoostRegimeModel(mapcode=country_code)
    training_file = f"{country_code}_full_data_2018_2024.csv"
    training_path = model.data_dir / training_file

    if not training_path.exists():
        return False

    xgb_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100,
        'random_state': 42
    }

    results = model.train(training_file=training_file, xgb_params=xgb_params)
    return bool(results)

def train_gru(country_code):
    trainer = GRUModelTrainer(mapcode=country_code, seq_len=168, pred_len=24)
    training_file = f"{country_code}_full_data_2018_2024.csv"
    training_path = trainer.data_dir / training_file

    if not training_path.exists():
        return False

    trainer.num_epochs = 50
    trainer.patience = 10

    try:
        trainer.train(train_file=str(training_path))
        trainer.save_model()
        return True
    except:
        return False

def train_informer(country_code):
    trainer = InformerModelTrainer(
        mapcode=country_code,
        seq_len=168,
        label_len=48,
        pred_len=24,
        batch_size=32,
        learning_rate=1e-4,
        epochs=50,
        early_stop_patience=10
    )
    training_file = f"{country_code}_full_data_2018_2024.csv"
    training_path = trainer.data_dir / training_file

    if not training_path.exists():
        return False

    try:
        results = trainer.train(data_path=training_path)
        return bool(results)
    except:
        return False

def train_models(country_code, model_type="all"):
    success = True

    if model_type.lower() in ["all", "xgboost"]:
        success &= train_xgboost(country_code)

    if model_type.lower() in ["all", "gru"]:
        success &= train_gru(country_code)

    if model_type.lower() in ["all", "informer"]:
        success &= train_informer(country_code)

    return success

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--country", "-c", nargs="+", default=["DK1"])
    parser.add_argument("--model", "-m", nargs="+", default=["all"])
    args = parser.parse_args()

    overall_success = True
    for country_code in args.country:
        for model_type in args.model:
            success = train_models(country_code, model_type)
            overall_success &= success

    if overall_success:
        print("All model training tasks completed successfully")
    else:
        print("Some model training tasks failed")