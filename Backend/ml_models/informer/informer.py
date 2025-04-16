import sys
import os

# Path to the informerModel directory
informer_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "informerModel"))

# Add informerModel/ to the Python path so Informer's own relative imports work
if informer_repo_root not in sys.path:
    sys.path.insert(0, informer_repo_root)


import torch
import json
from ml_models.informer.informerModel.models.model import Informer


class InformerWrapper:
    def __init__(self, config_path, weight_path, device="cpu"):
        with open(config_path) as f:
            full_config = json.load(f)

        # Rename pred_len → out_len for compatibility with Informer's constructor
        if "pred_len" in full_config:
            full_config["out_len"] = full_config.pop("pred_len")

        allowed_keys = {
            "enc_in", "dec_in", "c_out", "seq_len", "label_len", "out_len",
            "factor", "d_model", "n_heads", "e_layers", "d_layers", "d_ff",
            "dropout", "attn", "embed", "freq", "activation",
            "output_attention", "distil", "mix"
        }

        config = {k: v for k, v in full_config.items() if k in allowed_keys}
        self.config = config
        self.device = device

        self.model = Informer(**config).to(device)
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
