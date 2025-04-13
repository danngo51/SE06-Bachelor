import torch
from ml_models.informer.informer import InformerWrapper
from ml_models.gru.gru import GRUWrapper
from ml_models.preprocessing import load_csv

def run_pipeline():
    config_path = "ml_models/informer/config.json"
    weight_path = "ml_models/informer/informer_trained.pt"
    data_path = "ml_models/informer/test_input.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    informer = InformerWrapper(config_path, weight_path, device=device)
    gru = GRUWrapper(input_dim=512, hidden_dim=128, device=device)

    x_enc, x_mark_enc, x_dec, x_mark_dec = load_csv(data_path)
    x_enc, x_mark_enc, x_dec, x_mark_dec = (
        x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
    )

    embedding, prediction = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)
    gru_out = gru.run(embedding)

    return {
        "informer_embedding": embedding.squeeze().cpu().numpy().tolist(),
        "informer_prediction": prediction.squeeze().cpu().numpy().tolist(),
        "gru_output": gru_out.squeeze().cpu().numpy().tolist()
    }


