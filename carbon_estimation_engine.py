import csv
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

class CarbonEmissionEstimator:
    def __init__(self):
        self.feature_encoders = {}
        self.feature_names = []
        self.model_coefficients = None
        self.intercept = None
        self.is_trained = False
        self.scaler = StandardScaler()
        self.feature_stats = {}
        
    def load_and_prepare_data(self, csv_file_path):
        """Load CSV data and prepare for training with validation"""
        print("Loading and preparing data...")
        
        try:
            # Read data
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                data = list(reader)
            
            if not data:
                raise ValueError("Empty dataset!")
                
            print(f"Loaded {len(data)} records")
            
            # Validate column names
            required_columns = {'CarbonEmission'} | set(self._get_expected_features())
            actual_columns = set(data[0].keys())
            missing_columns = required_columns - actual_columns
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Separate features and target
            features = []
            targets = []
            
            # Get feature names (exclude target)
            self.feature_names = [col for col in data[0].keys() if col != 'CarbonEmission']
            
            # Calculate feature statistics
            self._initialize_feature_stats()
            
            for row in data:
                # Validate target
                try:
                    target = float(row['CarbonEmission'])
                    if target < 0:
                        warnings.warn(f"Negative emission value found: {target}")
                        continue
                    targets.append(target)
                except ValueError:
                    warnings.warn(f"Invalid emission value: {row['CarbonEmission']}")
                    continue
                
                # Extract and encode features
                feature_vector = []
                valid_row = True
                
                for feature_name in self.feature_names:
                    value = row[feature_name]
                    try:
                        encoded_value = self._encode_feature(feature_name, value)
                        feature_vector.append(encoded_value)
                        self._update_feature_stats(feature_name, encoded_value)
                    except ValueError as e:
                        warnings.warn(f"Error encoding {feature_name}: {str(e)}")
                        valid_row = False
                        break
                
                if valid_row:
                    features.append(feature_vector)
            
            features = np.array(features)
            targets = np.array(targets)
            
            if len(features) == 0:
                raise ValueError("No valid data rows after preprocessing!")
            
            # Scale features
            features = self.scaler.fit_transform(features)
            
            print(f"Successfully processed {len(features)} valid records")
            self._print_feature_stats()
            
            return features, targets
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    
    def _get_expected_features(self):
        """Return list of expected features"""
        return [
            'Body Type', 'Sex', 'Diet', 'How Often Shower',
            'Heating Energy Source', 'Transport', 'Vehicle Type',
            'Social Activity', 'Monthly Grocery Bill',
            'Frequency of Traveling by Air', 'Vehicle Monthly Distance Km',
            'Waste Bag Size', 'Waste Bag Weekly Count',
            'How Long TV PC Daily Hour', 'How Many New Clothes Monthly',
            'How Long Internet Daily Hour', 'Energy efficiency',
            'Recycling', 'Cooking_With'
        ]
    
    def _initialize_feature_stats(self):
        """Initialize statistics tracking for features"""
        self.feature_stats = {
            name: {
                'min': float('inf'),
                'max': float('-inf'),
                'sum': 0,
                'count': 0,
                'categories': set() if name not in [
                    'Monthly Grocery Bill', 'Vehicle Monthly Distance Km',
                    'Waste Bag Weekly Count', 'How Long TV PC Daily Hour',
                    'How Many New Clothes Monthly', 'How Long Internet Daily Hour'
                ] else None
            }
            for name in self.feature_names
        }
    
    def _update_feature_stats(self, feature_name, value):
        """Update statistics for a feature"""
        stats = self.feature_stats[feature_name]
        if isinstance(value, (int, float)):
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['sum'] += value
            stats['count'] += 1
        elif stats['categories'] is not None:
            stats['categories'].add(value)
            
    def _print_feature_stats(self):
        """Print feature statistics"""
        print("\n=== FEATURE STATISTICS ===")
        for name, stats in self.feature_stats.items():
            if stats['categories'] is not None:
                print(f"{name}: {len(stats['categories'])} unique categories")
            else:
                mean = stats['sum'] / stats['count'] if stats['count'] > 0 else 0
                print(f"{name}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={mean:.2f}")
    
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
    
    def train(self, X, y, n_folds=5):
        """Train the model using cross-validation"""
        print("Training carbon emission estimation model...")
        
        # Initialize cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Add intercept term
            X_train_int = np.column_stack([np.ones(X_train.shape[0]), X_train])
            X_val_int = np.column_stack([np.ones(X_val.shape[0]), X_val])
            
            try:
                # Train on this fold
                XtX = np.dot(X_train_int.T, X_train_int)
                XtX_inv = np.linalg.pinv(XtX)
                Xty = np.dot(X_train_int.T, y_train)
                coefficients = np.dot(XtX_inv, Xty)
                
                # Make predictions on validation set
                y_pred = np.dot(X_val_int, coefficients)
                
                # Calculate metrics
                cv_scores['mse'].append(mean_squared_error(y_val, y_pred))
                cv_scores['mae'].append(mean_absolute_error(y_val, y_pred))
                cv_scores['r2'].append(r2_score(y_val, y_pred))
                
                print(f"Fold {fold + 1}/{n_folds} - MSE: {cv_scores['mse'][-1]:.2f}, "
                      f"MAE: {cv_scores['mae'][-1]:.2f}, R²: {cv_scores['r2'][-1]:.4f}")
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        # Train final model on all data
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        try:
            XtX = np.dot(X_with_intercept.T, X_with_intercept)
            XtX_inv = np.linalg.pinv(XtX)
            Xty = np.dot(X_with_intercept.T, y)
            coefficients = np.dot(XtX_inv, Xty)
            
            self.intercept = coefficients[0]
            self.model_coefficients = coefficients[1:]
            self.is_trained = True
            
            # Print final results
            print("\n=== CROSS-VALIDATION RESULTS ===")
            print(f"Mean MSE: {np.mean(cv_scores['mse']):.2f} ± {np.std(cv_scores['mse']):.2f}")
            print(f"Mean MAE: {np.mean(cv_scores['mae']):.2f} ± {np.std(cv_scores['mae']):.2f}")
            print(f"Mean R²: {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
            
            return True
            
        except Exception as e:
            print(f"Final model training failed: {str(e)}")
            return False
    
    def predict(self, X):
        """Make predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        predictions = self.intercept + np.dot(X_scaled, self.model_coefficients)
        return predictions
    
    def predict_single_order(self, order_features):
        """Predict carbon emission for a single order with input validation"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Validate input features
        missing_features = set(self.feature_names) - set(order_features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Encode features
        encoded_features = []
        for feature_name in self.feature_names:
            value = order_features.get(feature_name, '')
            try:
                encoded_value = self._encode_feature_for_prediction(feature_name, value)
                encoded_features.append(encoded_value)
            except ValueError as e:
                raise ValueError(f"Error encoding {feature_name}: {str(e)}")
        
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
    
    def get_feature_importance(self):
        """Get feature importance based on standardized coefficients"""
        if not self.is_trained:
            return {}
        
        # Using standardized coefficients for feature importance
        importance = {}
        for i, feature_name in enumerate(self.feature_names):
            importance[feature_name] = abs(self.model_coefficients[i])
        
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, filepath):
        """Save trained model to file with additional metadata"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'feature_encoders': self.feature_encoders,
            'feature_names': self.feature_names,
            'model_coefficients': self.model_coefficients.tolist(),
            'intercept': self.intercept,
            'is_trained': self.is_trained,
            'feature_stats': {
                k: {
                    **{kk: vv for kk, vv in v.items() if kk != 'categories'},
                    'categories': list(v['categories']) if v.get('categories') is not None else None
                }
                for k, v in self.feature_stats.items()
            },
            'scaler_mean_': self.scaler.mean_.tolist() if self.is_trained else None,
            'scaler_scale_': self.scaler.scale_.tolist() if self.is_trained else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file with validation"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Validate model data
            required_keys = {'feature_encoders', 'feature_names', 'model_coefficients', 
                           'intercept', 'is_trained', 'scaler_mean_', 'scaler_scale_'}
            missing_keys = required_keys - set(model_data.keys())
            if missing_keys:
                raise ValueError(f"Missing required model data: {missing_keys}")
            
            self.feature_encoders = model_data['feature_encoders']
            self.feature_names = model_data['feature_names']
            self.model_coefficients = np.array(model_data['model_coefficients'])
            self.intercept = model_data['intercept']
            self.is_trained = model_data['is_trained']
            
            # Restore scaler
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(model_data['scaler_mean_'])
            self.scaler.scale_ = np.array(model_data['scaler_scale_'])
            
            # Restore feature statistics if available
            if 'feature_stats' in model_data:
                self.feature_stats = {
                    k: {
                        **{kk: vv for kk, vv in v.items() if kk != 'categories'},
                        'categories': set(v['categories']) if v.get('categories') is not None else None
                    }
                    for k, v in model_data['feature_stats'].items()
                }
            
            print(f"Model loaded from {filepath}")
            self._print_model_summary()
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def _print_model_summary(self):
        """Print a summary of the loaded model"""
        print("\n=== MODEL SUMMARY ===")
        print(f"Number of features: {len(self.feature_names)}")
        print("\nTop 5 most important features:")
        importance = self.get_feature_importance()
        for feature, score in list(importance.items())[:5]:
            print(f"  {feature}: {score:.4f}")
        
        # Print feature encoders summary
        print("\nFeature encoding summary:")
        for feature, encoder in self.feature_encoders.items():
            if isinstance(encoder, dict):
                n_categories = len(encoder)
                print(f"  {feature}: {n_categories} categories")

# Main execution
if __name__ == "__main__":
    print("=== CARBON EMISSION ESTIMATION ENGINE ===")
    print("Version 2.0 - Enhanced with cross-validation and feature scaling")
    
    try:
        # Initialize the estimator
        estimator = CarbonEmissionEstimator()
        
        # Load and prepare data
        X, y = estimator.load_and_prepare_data('data/Carbon Emission.csv')
        
        # Train the model with cross-validation
        success = estimator.train(X, y, n_folds=5)
        
        if success:
            # Show feature importance
            print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
            importance = estimator.get_feature_importance()
            print("\nTop 10 most influential factors:")
            for feature, score in list(importance.items())[:10]:
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
            
            try:
                predicted_emission = estimator.predict_single_order(sample_order)
                print(f"\nPredicted CO₂ emission for sample order: {predicted_emission:.2f} kg")
                print("\nMost influential features for this prediction:")
                importance = estimator.get_feature_importance()
                feature_values = []
                for feature, score in list(importance.items())[:5]:
                    value = sample_order.get(feature, 'N/A')
                    feature_values.append(f"{feature}: {value} (importance: {score:.4f})")
                print("\n".join(feature_values))
            except Exception as e:
                print(f"Prediction error: {str(e)}")
            
            print("\n✅ CARBON EMISSION ESTIMATION ENGINE READY!")
            print("Use estimator.predict_single_order(order_data) to make predictions")
            
        else:
            print("\n❌ Model training failed! Please check the data and try again.")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise
