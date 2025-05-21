import sys
import os

# Dynamically find the absolute path to the informerModel repo
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))  # Up from informer.py to Backend/
informer_model_path = os.path.join(project_root, 'ml_models', 'informer', 'informerModel')

if informer_model_path not in sys.path:
    sys.path.insert(0, informer_model_path)

import torch
import json
from models.model import Informer


class InformerWrapper:
    def __init__(self, config_path, weight_path, device="cpu", feature_dim=None):
        """
        Initialize the InformerWrapper with more robust defaults
        
        Args:
            config_path: Path to the Informer config file
            weight_path: Path to the Informer model weights
            device: Device to run the model on ('cpu' or 'cuda')
            feature_dim: Optional dimension of input features, will override config value if provided
        """        # Read the config file for dimensions
        try:
            with open(config_path) as f:
                full_config = json.load(f)
            
            # Rename pred_len → out_len for compatibility with Informer's constructor
            if "pred_len" in full_config:
                full_config["out_len"] = full_config.pop("pred_len")
                
            # Print the config details for debugging
            print(f"Loaded Informer config from {config_path}")
        except Exception as e:
            print(f"Error loading config file: {e}. Using default configuration.")
            # Set up default configuration if config file cannot be loaded
            full_config = {}
        
        # Define default values
        default_config = {
            "enc_in": feature_dim if feature_dim else 7,  # Default number of input features
            "dec_in": feature_dim if feature_dim else 7,
            "c_out": 1,  # Default number of output features
            "seq_len": 168,  # Default sequence length (1 week)
            "label_len": 24,  # Default label length
            "out_len": 24,  # Default prediction length (1 day)
            "factor": 5,  # Default factor
            "d_model": 512,  # Default dimension of the model
            "n_heads": 8,  # Default number of heads
            "e_layers": 2,  # Default number of encoder layers
            "d_layers": 1,  # Default number of decoder layers
            "d_ff": 2048,  # Default feed forward dimension
            "dropout": 0.05,  # Default dropout rate
            "attn": "prob",  # Default attention type
            "embed": "timeF",  # Default embedding
            "freq": "h",  # Default frequency (hourly)
            "activation": "gelu",  # Default activation function
            "output_attention": False,  # Default output attention
            "distil": True,  # Default distillation
            "mix": True  # Default mix
        }
        
        # Merge default with loaded config
        for key, default_value in default_config.items():
            if key not in full_config:
                full_config[key] = default_value
        
        # Preserve the trained model structure if we're loading weights
        try:
            # Get model architecture from saved weights
            saved_state = torch.load(weight_path, map_location=device)
            
            # Check for encoder architecture
            if "encoder.attn_layers.0.conv1.weight" in saved_state:
                enc_weight = saved_state["encoder.attn_layers.0.conv1.weight"]
                if len(enc_weight.shape) > 1:
                    # Extract d_ff from encoder conv1 weight shape
                    d_ff = enc_weight.shape[0]
                    print(f"Detected d_ff={d_ff} from saved weights")
                    full_config["d_ff"] = d_ff
            
            # Check for input feature dimension
            if "enc_embedding.value_embedding.tokenConv.weight" in saved_state:
                enc_weight = saved_state["enc_embedding.value_embedding.tokenConv.weight"]
                if len(enc_weight.shape) > 1:
                    # Extract enc_in from encoder embedding weight shape
                    enc_in = enc_weight.shape[1]
                    print(f"Detected enc_in={enc_in} from saved weights")
                    full_config["enc_in"] = enc_in
                    full_config["dec_in"] = enc_in
            
            # Check if we need to force feature_dim to match the trained model
            if feature_dim and feature_dim != full_config["enc_in"]:
                print(f"WARNING: Input data has {feature_dim} features but model was trained with {full_config['enc_in']} features")
                print(f"Consider preparing your data to have exactly {full_config['enc_in']} features")
                
                # We'll use the model's dimensions since we want to load the weights correctly
                feature_dim = full_config["enc_in"]
        except Exception as e:
            print(f"Could not analyze model weights: {e}. Using provided configuration.")
            
            # Override with feature_dim if provided
            if feature_dim is not None:
                print(f"Overriding feature dimensions with provided value: {feature_dim}")
                full_config["enc_in"] = feature_dim
                full_config["dec_in"] = feature_dim
        
        allowed_keys = {
            "enc_in", "dec_in", "c_out", "seq_len", "label_len", "out_len",
            "factor", "d_model", "n_heads", "e_layers", "d_layers", "d_ff",
            "dropout", "attn", "embed", "freq", "activation",
            "output_attention", "distil", "mix"
        }
        
        # Filter only allowed keys
        config = {k: v for k, v in full_config.items() if k in allowed_keys}
        
        # Log the config being used
        print(f"Using Informer configuration: {config}")
        
        self.config = config
        self.device = device

        try:
            self.model = Informer(**config).to(device)
            self.model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
            self.model.eval()
            print(f"✅ Informer model loaded successfully from {weight_path}")
        except Exception as e:
            print(f"Error loading Informer model: {e}. Creating a new model with default parameters.")
            self.model = Informer(**config).to(device)
            self.model.eval()
            print("⚠️ Using untrained Informer model.")
        
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
        
    def encode(self, x_enc, x_mark_enc):
        """
        Get only the encoder output (embeddings) from the Informer model.
        Useful for training the GRU model.
        
        Args:
            x_enc: Encoder input tensor [batch_size, seq_len, feature_dim]
            x_mark_enc: Encoder time feature tensor [batch_size, seq_len, time_feature_dim]
            
        Returns:
            Encoder output tensor [batch_size, seq_len, d_model]
        """
        with torch.no_grad():
            # Move inputs to device if they're not already there
            if x_enc.device != self.device:
                x_enc = x_enc.to(self.device)
            if x_mark_enc.device != self.device:
                x_mark_enc = x_mark_enc.to(self.device)
                
            # Create dummy decoder inputs just to run the forward pass
            batch_size, seq_len = x_enc.shape[0], x_enc.shape[1]
            pred_len = self.config.get("out_len", 24)
            label_len = self.config.get("label_len", 24)
            
            x_dec = torch.zeros((batch_size, label_len + pred_len, x_enc.shape[2]), device=self.device)
            x_mark_dec = torch.zeros((batch_size, label_len + pred_len, x_mark_enc.shape[2]), device=self.device)
              # Run through model and return only encoder output
            enc_out = self.model.enc_embedding(x_enc, x_mark_enc)
            enc_out, _ = self.model.encoder(enc_out)
            
            return enc_out
