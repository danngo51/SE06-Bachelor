#!/usr/bin/env python
"""
Test script for the restructured hybrid_model.py
This script tests prediction with the zone-specific hybrid model
"""
import os
import sys
from pathlib import Path

# Add the Backend directory to the path
current_file_path = Path(os.path.abspath(__file__))
backend_path = current_file_path.parent.parent
sys.path.insert(0, str(backend_path))

from ml_models.hybrid_model import predict_with_hybrid_model, test_predict
import argparse

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the restructured hybrid model")
    parser.add_argument('--zone', type=str, default="DK1", help="Zone code (e.g., DK1, SE1)")
    parser.add_argument('--date', type=str, default="2025-05-01", help="Prediction date (YYYY-MM-DD)")
    parser.add_argument('--test-mode', action="store_true", help="Use test mode instead of real model")
    args = parser.parse_args()
    print(f"🚀 Testing hybrid model for zone {args.zone} on date {args.date}")
    
    try:
        # Set up paths for easy testing
        current_file_path = Path(os.path.abspath(__file__))
        ml_models_dir = current_file_path.parent
        
        if args.test_mode:
            print(f"ℹ️ Using test mode...")
            result = test_predict(args.zone, args.date)
        else:
            print(f"ℹ️ Using real model...")
            # Setup paths explicitly for testing
            data_dir = ml_models_dir / "data"
            informer_dir = ml_models_dir / "informer"
            gru_dir = ml_models_dir / "gru"
            
            input_file_path = str(data_dir / args.zone / "prediction_data.csv")
            if not os.path.exists(input_file_path):
                input_file_path = str(data_dir / "DK1" / "prediction_data.csv")
                print(f"⚠️ Using fallback input data from DK1")
                
            config_path = str(informer_dir / args.zone / "config.json")
            if not os.path.exists(config_path):
                config_path = str(informer_dir / "config.json")
                print(f"⚠️ Using fallback config")
                
            weight_path = str(informer_dir / args.zone / "results" / "checkpoint.pth")
            if not os.path.exists(weight_path):
                weight_path = str(informer_dir / "results" / "checkpoint.pth")
                print(f"⚠️ Using fallback Informer weights")
                
            gru_path = str(gru_dir / args.zone / "results" / "gru_trained.pt")
            if not os.path.exists(gru_path):
                gru_path = str(gru_dir / "results" / "gru_trained.pt")
                print(f"⚠️ Using fallback GRU weights")
            
            print(f"📁 Using input file: {input_file_path}")
            print(f"📁 Using config: {config_path}")
            print(f"📁 Using Informer weights: {weight_path}")
            print(f"📁 Using GRU weights: {gru_path}")
            
            result = predict_with_hybrid_model(
                args.zone, 
                args.date,
                input_file_path=input_file_path,
                config_path=config_path,
                weight_path=weight_path,
                gru_path=gru_path
            )
        
        # Print results
        print("\n📊 Prediction Results:")
        print(f"✅ Informer prediction: {result.informer_prediction[:5]}... (showing first 5 hours)")
        print(f"✅ GRU prediction: {result.gru_prediction[:5]}... (showing first 5 hours)")
        print(f"✅ Combined model prediction: {result.model_prediction[:5]}... (showing first 5 hours)")
        
        print("\n🏁 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
