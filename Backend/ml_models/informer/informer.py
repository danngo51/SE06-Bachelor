import torch
import json
from informerModel.models.model import Informer

class InformerWrapper:
    def __init__(self, config_path, weight_path, device="cpu"):
        with open(config_path) as f:
            config = json.load(f)
        self.model = Informer(**config).to(device)
        self.device = device
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.eval()

    def run(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        with torch.no_grad():
            enc_out, pred = self.model(
                x_enc.to(self.device),
                x_mark_enc.to(self.device),
                x_dec.to(self.device),
                x_mark_dec.to(self.device),
                return_enc_and_pred=True
            )
        return enc_out, pred
