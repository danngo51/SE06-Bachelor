# File: ml_models/gru/gruModel/training_gru.py

import torch
import torch.nn as nn
import torch.optim as optim
from informer.informer import InformerWrapper
from gruModel.gruModel import GRUModel
from preprocessing import load_training_batch  # You should implement this
import json
import os

# --- Config ---
CONFIG_PATH = "../../informer/config.json"
WEIGHT_PATH = "../../informer/informer_trained.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
LR = 1e-3
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load Frozen Informer ---
informer = InformerWrapper(CONFIG_PATH, WEIGHT_PATH, device=DEVICE)
for p in informer.model.parameters():
    p.requires_grad = False

# --- Load GRU Model and Head ---
gru_model = GRUModel(input_dim=512, hidden_dim=128).to(DEVICE)
regressor = nn.Linear(128, 1).to(DEVICE)

# --- Optimizer and Loss ---
optimizer = optim.Adam(list(gru_model.parameters()) + list(regressor.parameters()), lr=LR)
criterion = nn.MSELoss()

# --- Training Loop ---
for epoch in range(EPOCHS):
    epoch_loss = 0
    for x_enc, x_mark_enc, x_dec, x_mark_dec, y_true in load_training_batch():  # implement batch loader
        x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = \
            x_enc.to(DEVICE), x_mark_enc.to(DEVICE), x_dec.to(DEVICE), x_mark_dec.to(DEVICE), y_true.to(DEVICE)

        with torch.no_grad():
            encoder_out, _ = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)

        gru_out = gru_model(encoder_out)
        pred = regressor(gru_out)

        loss = criterion(pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

# --- Save Models ---
torch.save(gru_model.state_dict(), os.path.join(SAVE_DIR, "gru_trained.pt"))
torch.save(regressor.state_dict(), os.path.join(SAVE_DIR, "gru_regressor.pt"))
