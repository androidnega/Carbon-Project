import numpy as np
from scipy import stats
import csv

# Feature encoding dictionaries
body_type_map = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
sex_map = {'male': 0, 'female': 1}
diet_map = {'omnivore': 0, 'vegetarian': 1, 'vegan': 2, 'pescatarian': 3}
transport_map = {'walk/bicycle': 0, 'public': 1, 'private': 2}

def encode_features(row):
    features = []
    
    # Encode categorical variables
    features.append(body_type_map.get(row['Body Type'].lower(), 1))  # default to normal
    features.append(sex_map.get(row['Sex'].lower(), 0))
    features.append(diet_map.get(row['Diet'].lower(), 0))
    features.append(transport_map.get(row['Transport'].lower(), 0))
    
    # Numeric features
    try:
        features.append(float(row['Monthly Grocery Bill']))
        features.append(float(row['Vehicle Monthly Distance Km']))
        features.append(float(row['How Long TV PC Daily Hour']))
    except ValueError:
        return None
    
    return features

def load_data():
    features_list = []
    emissions = []
    
    with open('data/Carbon Emission.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                features = encode_features(row)
                if features is not None:
                    emission = float(row['CarbonEmission'])
                    features_list.append(features)
                    emissions.append(emission)
            except (ValueError, KeyError):
                continue
    
    return np.array(features_list), np.array(emissions)

def train_models(X, y):
    """Train separate models for each feature and return the best ones"""
    models = []
    feature_names = [
        'Body Type', 'Sex', 'Diet', 'Transport',
        'Monthly Grocery Bill', 'Vehicle Monthly Distance',
        'TV/PC Hours'
    ]
    
    print("\nTraining individual feature models:")
    print("-" * 50)
    
    for i in range(X.shape[1]):
        slope, intercept, r_value, p_value, std_err = stats.linregress(X[:, i], y)
        r_squared = r_value ** 2
        models.append({
            'feature': feature_names[i],
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err
        })
        print(f"{feature_names[i]}:")
        print(f"  R-squared: {r_squared:.4f}")
        print(f"  P-value: {p_value:.4f}")
    
    # Sort models by R-squared value
    models.sort(key=lambda x: x['r_squared'], reverse=True)
    return models

def predict_emission(features, models):
    """Make a prediction using weighted average of top models"""
    prediction = 0
    weight_sum = 0
    
    # Use top 3 models for prediction
    for model in models[:3]:
        weight = model['r_squared']
        feature_idx = [
            'Body Type', 'Sex', 'Diet', 'Transport',
            'Monthly Grocery Bill', 'Vehicle Monthly Distance',
            'TV/PC Hours'
        ].index(model['feature'])
        
        pred = model['slope'] * features[feature_idx] + model['intercept']
        prediction += pred * weight
        weight_sum += weight
    
    return prediction / weight_sum if weight_sum > 0 else 0

def main():
    print("Loading and processing data...")
    X, y = load_data()
    
    if len(X) == 0:
        print("Error: No valid data found!")
        return
    
    print(f"\nDataset size: {len(X)} samples")
    
    print("\nTraining models...")
    models = train_models(X, y)
    
    print("\nTop 3 most influential factors:")
    for i, model in enumerate(models[:3], 1):
        print(f"\n{i}. {model['feature']}")
        print(f"   R-squared: {model['r_squared']:.4f}")
        print(f"   Equation: emission = {model['slope']:.4f} * {model['feature']} + {model['intercept']:.4f}")
    
    # Example predictions
    print("\nSample predictions:")
    example_features = [
        [2, 0, 0, 1, 200, 1000, 4],  # High carbon example
        [1, 1, 2, 0, 100, 0, 2]      # Low carbon example
    ]
    
    for features in example_features:
        prediction = predict_emission(features, models)
        print(f"\nPrediction for:")
        print(f"- Body Type: {'overweight' if features[0] == 2 else 'normal'}")
        print(f"- Sex: {'male' if features[1] == 0 else 'female'}")
        print(f"- Diet: {list(diet_map.keys())[list(diet_map.values()).index(features[2])]}")
        print(f"- Transport: {list(transport_map.keys())[list(transport_map.values()).index(features[3])]}")
        print(f"- Monthly Grocery Bill: ${features[4]}")
        print(f"- Vehicle Monthly Distance: {features[5]}km")
        print(f"- TV/PC Hours Daily: {features[6]}")
        print(f"Estimated Carbon Emission: {prediction:.2f} units")

if __name__ == "__main__":
    main()
