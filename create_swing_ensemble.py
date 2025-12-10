"""
Quick script to create swing_ensemble.pkl files from existing regime_models.pkl.
This bridges the gap between train_model.py (saves regime_models.pkl) and
multi_horizon_ensemble.py (expects swing_ensemble.pkl).
"""
import joblib
from pathlib import Path

def create_swing_ensemble(exchange: str):
    model_dir = Path('models') / exchange
    
    if not (model_dir / 'regime_models.pkl').exists():
        print(f"ERROR: {model_dir}/regime_models.pkl not found")
        return False
        
    # Load the pre-trained regime models
    regime_models = joblib.load(model_dir / 'regime_models.pkl')
    
    # Use the first regime model as the swing model
    # (In a proper setup, you'd train a dedicated swing model)
    first_model = list(regime_models.values())[0]
    
    # Create the swing ensemble data structure
    swing_ensemble_data = {
        'models': {'lightgbm': first_model},
        'weights': {'lightgbm': 1.0},
        '_is_fitted': True
    }
    
    # Save it
    joblib.dump(swing_ensemble_data, model_dir / 'swing_ensemble.pkl')
    print(f"✓ Created {model_dir}/swing_ensemble.pkl")
    return True

if __name__ == '__main__':
    for exchange in ['NSE', 'BSE']:
        create_swing_ensemble(exchange)
    print("Done! Restart the OI Tracker to use the new models.")
