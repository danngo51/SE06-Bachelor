import torch.nn as nn

from .informerModel.models.model import Informer

class InformerWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.informer = Informer(
            enc_in=config["enc_in"],
            dec_in=config["dec_in"],
            c_out=config["c_out"],
            seq_len=config["seq_len"],
            label_len=config["label_len"],
            out_len=config["pred_len"],
            factor=config["factor"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            e_layers=config["e_layers"],
            d_layers=config["d_layers"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            attn=config["attn"],
            embed=config["embed"],
            freq=config["freq"],
            activation=config["activation"],
            output_attention=config["output_attention"],
            distil=config["distil"],
            mix=config["mix"],
            device=config["device"]
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Input shapes:
        - x_enc:       (batch, seq_len, input_dim)
        - x_mark_enc:  (batch, seq_len, time_features)
        - x_dec:       (batch, pred_len, input_dim)
        - x_mark_dec:  (batch, pred_len, time_features)

        Output:
        - (batch, pred_len, c_out) or encoder output, depending on Informer config
        """
        return self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec)