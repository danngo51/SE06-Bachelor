import torch
from informer.informer import InformerWrapper
from gruModel.gruModel import GRUModel
from preprocessing import load_input_sample

# Load model
gru = GRUModel()
gru.load_state_dict(torch.load("ml_models/gru/results/gru_trained.pt"))
gru.eval()

informer = InformerWrapper(
    config_path="ml_models/informer/config.json",
    weight_path="ml_models/informer/results/checkpoint.pth"
)
informer.model.eval()

# Load a 168-hour input sample from 2024
x_enc, x_mark_enc, x_dec, x_mark_dec = load_input_sample("some_2024_window.csv")

with torch.no_grad():
    enc_out, _ = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)
    pred = gru(enc_out)

print("GRU prediction:", pred.numpy())
