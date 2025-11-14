# fix_model.py
import pickle
import numpy as np

def fix_model_compatibility():
    try:
        # Load the problematic model
        with open('bigmart_best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"Original model type: {type(model_data)}")
        
        # Create a compatible model structure
        fixed_model = {
            'model': model_data,
            'metadata': {
                'fixed_for_compatibility': True,
                'original_type': str(type(model_data))
            }
        }
        
        # Save the fixed model
        with open('bigmart_best_model_fixed.pkl', 'wb') as f:
            pickle.dump(fixed_model, f)
        
        print("✓ Fixed model saved as 'bigmart_best_model_fixed.pkl'")
        return True
        
    except Exception as e:
        print(f"Error fixing model: {e}")
        return False

if __name__ == "__main__":
    fix_model_compatibility()
