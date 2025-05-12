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
        if args.test_mode:
            print(f"ℹ️ Using test mode...")
            result = test_predict(args.zone, args.date)
        else:
            print(f"ℹ️ Using real model...")
            result = predict_with_hybrid_model(args.zone, args.date)
        
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
