import torch
import torch.nn as nn
import torch.optim as optim

from informer.informer import InformerWrapper
from gruModel.gruModel import GRUModel
from preprocessing import load_training_batch

# --- Load Informer ---
informer = InformerWrapper(
    config_path="ml_models/informer/config.json",
    weight_path="ml_models/informer/results/checkpoint.pth"
)
informer.model.eval()

# --- GRU model ---
gru = GRUModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(gru.parameters(), lr=0.001)

# --- Training loop ---
for epoch in range(10):
    total_loss = 0
    for x_enc, x_mark_enc, x_dec, x_mark_dec, y in load_training_batch():
        with torch.no_grad():
            enc_out, _ = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, 168, 512]

        pred = gru(enc_out)  # [B, 24]
        loss = criterion(pred, y)  # y should be [B, 24]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

# --- Save GRU weights ---
torch.save(gru.state_dict(), "ml_models/gru/results/gru_trained.pt")
