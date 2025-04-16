import torch
import torch.nn as nn
import torch.optim as optim

from ml_models.informer.informer import InformerWrapper
from ml_models.gru.gruModel.gruModel import GRUModel
from ml_models.preprocessing import load_training_batch

def main():
    # ✅ 1. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ✅ 2. Load Informer (frozen)
    config_path = "ml_models/informer/config.json"
    weight_path = "ml_models/informer/results/checkpoint.pth"
    informer = InformerWrapper(config_path, weight_path, device=device)
    informer.model.eval()

    # ✅ 3. Init GRU + Regressor on correct device
    gru = GRUModel(input_dim=512, hidden_dim=128, output_dim=128).to(device)
    regressor = nn.Linear(128, 24).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(gru.parameters()) + list(regressor.parameters()), lr=0.001)

    # ✅ 4. Training loop
    EPOCHS = 3
    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0

        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in load_training_batch():
            # Move data to device
            x_enc = x_enc.to(device)
            x_mark_enc = x_mark_enc.to(device)
            x_dec = x_dec.to(device)
            x_mark_dec = x_mark_dec.to(device)
            y = y.to(device)

            # Get encoder output
            with torch.no_grad():
                enc_out, _ = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # GRU + prediction
            gru_out = gru(enc_out)            # [B, 128]
            pred = regressor(gru_out)         # [B, 24]

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")

    # ✅ 5. Save trained models
    torch.save(gru.state_dict(), "ml_models/gru/results/gru_trained.pt")
    torch.save(regressor.state_dict(), "ml_models/gru/results/gru_regressor.pt")
    print("✅ GRU and regressor models saved.")

if __name__ == "__main__":
    main()
