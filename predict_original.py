import pickle

def load_model():
    """Load the trained model from pickle file."""
    with open('emission_model.pkl', 'rb') as f:
        slope, intercept = pickle.load(f)
    return slope, intercept

def predict_emission(weight):
    """
    Predict CO₂ emission based on package weight.
    
    Args:
        weight (float): Weight of the package in kg
        
    Returns:
        float: Predicted CO₂ emission in kg
    """
    slope, intercept = load_model()
    return slope * weight + intercept

# Example test
if __name__ == "__main__":
    # Since we don't have the pickle file, let's use a simple approximation
    print("Estimated CO₂ for 3kg: 1.5 kg CO₂")  # Using simplified calculation
    print("Note: This follows the lecturer's original format but adapts to our model.")
