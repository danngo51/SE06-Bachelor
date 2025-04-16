import sys
import os

# Step 1: Make sure Python sees your repo root as the base for imports
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Step 2: Import from your GRU trainer (and it will internally load InformerWrapper)
from ml_models.gru.gruModel import gruTrain

# Step 3: Call your training entry point
if __name__ == "__main__":
    gruTrain.main()  # Assuming your gruTrain.py has a main() function
