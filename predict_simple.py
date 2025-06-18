import json
import numpy as np

def load_model():
    """Load the trained regression model parameters from JSON file."""
    try:
        with open('carbon_emission_model.json', 'r') as f:
            model_params = json.load(f)
        return model_params
    except FileNotFoundError:
        raise Exception("Model file 'carbon_emission_model.json' not found. Please train the model first.")

def predict_emission(weight):
    """
    Predict CO₂ emission based on package weight (simplified version).
    
    Args:
        weight (float): Weight of the package in kg
        
    Returns:
        float: Predicted CO₂ emission in kg
    """
    # For simplicity, use a basic linear relationship
    # Real implementation would use the full feature set from the trained model
    base_emission = 0.5  # Base CO2 emission per kg
    return weight * base_emission

if __name__ == "__main__":
    # Example test
    print("Estimated CO₂ for 3kg:", predict_emission(3))
    print("Estimated CO₂ for 5kg:", predict_emission(5))
    print("Estimated CO₂ for 10kg:", predict_emission(10))
    print("\nNote: This is a simplified version. The actual model uses 19 features.")
    print("For complete predictions, use the full feature set from the carbon dataset.")
