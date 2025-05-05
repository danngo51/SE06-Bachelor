import torch
import os

# Import the hybrid model predictor
from ml_models.hybrid_model import predict_with_hybrid_model

def main():
    # Set your fixed input and output paths here
    input_path = "ml_models/data/pred_input_test.csv"      # <<< Set your test input file path
    output_path = "ml_models/results/pred_output_test.csv"     # <<< Set your test output file path

    # Check input exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Running on device: {device}")

    # Run the hybrid prediction
    predict_with_hybrid_model(input_path, output_path)

if __name__ == "__main__":
    main()
