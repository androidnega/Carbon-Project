import csv
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import json

class CarbonEmissionEstimator:
    def __init__(self):
        self.feature_encoders = {}
        self.feature_names = []
        self.model_coefficients = None
        self.intercept = None
        self.is_trained = False
        
    def load_and_prepare_data(self, csv_file_path):
        """Load CSV data and prepare for training"""
        print("Loading and preparing data...")
        
        # Read data
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = list(reader)
        
        print(f"Loaded {len(data)} records")
        
        # Separate features and target
        features = []
        targets = []
        
        # Get feature names (exclude target)
        self.feature_names = [col for col in data[0].keys() if col != 'CarbonEmission']
        
        for row in data:
            # Extract target
            target = float(row['CarbonEmission'])
            targets.append(target)
            
            # Extract and encode features
            feature_vector = []
            for feature_name in self.feature_names:
                value = row[feature_name]
                encoded_value = self._encode_feature(feature_name, value)
                feature_vector.append(encoded_value)
            
            features.append(feature_vector)
        
        return np.array(features), np.array(targets)
    
    def _encode_feature(self, feature_name, value):
        """Encode categorical features to numerical values"""
        if feature_name not in self.feature_encoders:
            self.feature_encoders[feature_name] = {}
        
        # Handle numerical features
        if feature_name in ['Monthly Grocery Bill', 'Vehicle Monthly Distance Km', 
                          'Waste Bag Weekly Count', 'How Long TV PC Daily Hour',
                          'How Many New Clothes Monthly', 'How Long Internet Daily Hour']:
            try:
                return float(value) if value else 0.0
            except:
                return 0.0
        
        # Handle categorical features
        if value not in self.feature_encoders[feature_name]:
            # Assign next available integer
            next_id = len(self.feature_encoders[feature_name])
            self.feature_encoders[feature_name][value] = next_id
        
        return float(self.feature_encoders[feature_name][value])
    
    def train(self, X, y):
        """Train the linear regression model using scipy"""
        print("Training carbon emission estimation model...")
        
        # Add intercept term (bias)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Simple linear regression using normal equation
        # θ = (X^T X)^(-1) X^T y
        try:
            XtX = np.dot(X_with_intercept.T, X_with_intercept)
            XtX_inv = np.linalg.pinv(XtX)  # Use pseudo-inverse for stability
            Xty = np.dot(X_with_intercept.T, y)
            coefficients = np.dot(XtX_inv, Xty)
            
            self.intercept = coefficients[0]
            self.model_coefficients = coefficients[1:]
            self.is_trained = True
            
            # Calculate training metrics
            predictions = self.predict(X)
            mse = np.mean((predictions - y) ** 2)
            r2 = self._calculate_r2(y, predictions)
            
            print(f"Training completed!")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"R² Score: {r2:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def predict(self, X):
        """Make predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        predictions = self.intercept + np.dot(X, self.model_coefficients)
        return predictions
    
    def predict_single_order(self, order_features):
        """Predict carbon emission for a single e-commerce order"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Encode the order features
        encoded_features = []
        for feature_name in self.feature_names:
            value = order_features.get(feature_name, '')
            encoded_value = self._encode_feature_for_prediction(feature_name, value)
            encoded_features.append(encoded_value)
        
        # Make prediction
        X = np.array([encoded_features])
        prediction = self.predict(X)[0]
        return max(0, prediction)  # Ensure non-negative emission
    
    def _encode_feature_for_prediction(self, feature_name, value):
        """Encode feature for prediction (handle unseen categories)"""
        if feature_name in ['Monthly Grocery Bill', 'Vehicle Monthly Distance Km', 
                          'Waste Bag Weekly Count', 'How Long TV PC Daily Hour',
                          'How Many New Clothes Monthly', 'How Long Internet Daily Hour']:
            try:
                return float(value) if value else 0.0
            except:
                return 0.0
        
        # For categorical features, use existing encoding or default to 0
        if feature_name in self.feature_encoders and value in self.feature_encoders[feature_name]:
            return float(self.feature_encoders[feature_name][value])
        else:
            return 0.0  # Default for unseen categories
    
    def _calculate_r2(self, y_true, y_pred):
        """Calculate R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_feature_importance(self):
        """Get feature importance based on coefficient magnitudes"""
        if not self.is_trained:
            return {}
        
        importance = {}
        for i, feature_name in enumerate(self.feature_names):
            importance[feature_name] = abs(self.model_coefficients[i])
        
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'feature_encoders': self.feature_encoders,
            'feature_names': self.feature_names,
            'model_coefficients': self.model_coefficients.tolist(),
            'intercept': self.intercept,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.feature_encoders = model_data['feature_encoders']
        self.feature_names = model_data['feature_names']
        self.model_coefficients = np.array(model_data['model_coefficients'])
        self.intercept = model_data['intercept']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")

# Main execution
if __name__ == "__main__":
    print("=== PHASE 1: CARBON EMISSION ESTIMATION ENGINE ===")
    
    # Initialize the estimator
    estimator = CarbonEmissionEstimator()
    
    # Load and prepare data
    X, y = estimator.load_and_prepare_data('data/Carbon Emission.csv')
    
    # Train the model
    success = estimator.train(X, y)
    
    if success:
        # Show feature importance
        print("\n=== FEATURE IMPORTANCE ===")
        importance = estimator.get_feature_importance()
        for feature, score in list(importance.items())[:10]:  # Top 10
            print(f"{feature}: {score:.4f}")
        
        # Save the trained model
        estimator.save_model('carbon_emission_model.json')
        
        # Test with a sample order
        print("\n=== SAMPLE PREDICTION ===")
        sample_order = {
            'Body Type': 'normal',
            'Sex': 'male', 
            'Diet': 'omnivore',
            'How Often Shower': 'daily',
            'Heating Energy Source': 'electricity',
            'Transport': 'private',
            'Vehicle Type': 'petrol',
            'Social Activity': 'often',
            'Monthly Grocery Bill': '200',
            'Frequency of Traveling by Air': 'rarely',
            'Vehicle Monthly Distance Km': '1000',
            'Waste Bag Size': 'medium',
            'Waste Bag Weekly Count': '2',
            'How Long TV PC Daily Hour': '5',
            'How Many New Clothes Monthly': '10',
            'How Long Internet Daily Hour': '4',
            'Energy efficiency': 'Sometimes',
            'Recycling': "['Paper', 'Plastic']",
            'Cooking_With': "['Stove', 'Oven']"
        }
        
        predicted_emission = estimator.predict_single_order(sample_order)
        print(f"Predicted CO₂ emission for sample order: {predicted_emission:.2f} kg")
        
        print("\n✅ PHASE 1 COMPLETED: AI Model Ready!")
    else:
        print("❌ Model training failed!")
