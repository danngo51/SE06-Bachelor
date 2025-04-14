# File: ml_models/gru/gruModel/predict_gru.py

import torch
from informer.informer import InformerWrapper
from gruModel.gruModel import GRUModel
from preprocessing import load_input_sample  # you write this
import torch.nn as nn
import json

# --- Config ---
CONFIG_PATH = "../../informer/config.json"
WEIGHT_PATH = "../../informer/informer_trained.pt"
GRU_PATH = "checkpoints/gru_trained.pt"
HEAD_PATH = "checkpoints/gru_regressor.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Frozen Informer ---
informer = InformerWrapper(CONFIG_PATH, WEIGHT_PATH, device=DEVICE)
informer.model.eval()

# --- Load Trained GRU + Head ---
gru = GRUModel(input_dim=512, hidden_dim=128).to(DEVICE)
gru.load_state_dict(torch.load(GRU_PATH, map_location=DEVICE))
gru.eval()

regressor = nn.Linear(128, 1).to(DEVICE)
regressor.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE))
regressor.eval()

# --- Load 168h Input from 2024 ---
x_enc, x_mark_enc, x_dec, x_mark_dec = load_input_sample("input_2024.csv")  # implement this
x_enc, x_mark_enc, x_dec, x_mark_dec = (
    x_enc.to(DEVICE), x_mark_enc.to(DEVICE), x_dec.to(DEVICE), x_mark_dec.to(DEVICE)
)

# --- Inference ---
with torch.no_grad():
    encoder_out, _ = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)
    gru_out = gru(encoder_out)
    pred = regressor(gru_out)

print(f"📈 Predicted price: {pred.item():.2f}")
