import json

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
    Predict CO₂ emission based on package weight.
    
    Args:
        weight (float): Weight of the package in kg
        
    Returns:
        float: Predicted CO₂ emission in kg
    """
    if not isinstance(weight, (int, float)) or weight <= 0:
        raise ValueError("Weight must be a positive number")    # For this simplified version, we use a basic linear relationship
    # In a real implementation, we would use the trained model coefficients
    base_rate = 0.5  # kg CO2 per kg package weight
    return weight * base_rate

if __name__ == "__main__":
    # Example test cases
    test_weights = [3, 5, 10]
    print("CO₂ Emission Predictions:")
    print("-" * 40)
    print("Weight (kg) | CO₂ Emission (kg)")
    print("-" * 40)
    
    for weight in test_weights:
        try:
            emission = predict_emission(weight)
            print(f"{weight:^10.1f} | {emission:^15.2f}")
        except Exception as e:
            print(f"Error for {weight}kg: {str(e)}")
    print("-" * 40)
