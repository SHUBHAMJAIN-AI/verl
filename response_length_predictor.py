#!/usr/bin/env python3
"""
Practical Response Length Predictor

A simple, deployable model for predicting response lengths based on prompt characteristics.
Trained on VERL PPO training data.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class ResponseLengthPredictor:
    """
    A trained model for predicting response lengths from prompt features.
    
    Features used:
    - prompt_length: Length of input prompt in tokens
    - training_step: Training step (for models in training)
    - sequence_score: Quality score of the prompt/response pair
    - unique_tokens_ratio: Diversity of tokens in prompt
    - repetition_ratio: Amount of repetition in prompt
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'prompt_length', 'training_step', 'step_normalized',
            'unique_tokens_ratio', 'repetition_ratio', 'sequence_score',
            'sequence_reward', 'prompt_length_squared', 'prompt_length_log',
            'prompt_step_interaction'
        ]
        self.is_trained = False
        
    def calculate_prompt_features(self, prompt_length, training_step=350, 
                                sequence_score=0.7, max_step=350):
        """
        Calculate features from basic prompt characteristics.
        
        Args:
            prompt_length (int): Length of prompt in tokens
            training_step (int): Current training step (default: 350)
            sequence_score (float): Quality score 0-1 (default: 0.7)
            max_step (int): Maximum training step for normalization
            
        Returns:
            dict: Calculated features
        """
        # Simulate token analysis (in practice, would analyze actual tokens)
        unique_tokens_ratio = min(0.95, 0.3 + (prompt_length / 500))  # Longer prompts tend to be more diverse
        repetition_ratio = max(0.05, 0.4 - (prompt_length / 200))      # Shorter prompts tend to repeat more
        
        step_normalized = training_step / max_step
        prompt_length_squared = prompt_length ** 2
        prompt_length_log = np.log1p(prompt_length)
        prompt_step_interaction = prompt_length * step_normalized
        
        return {
            'prompt_length': prompt_length,
            'training_step': training_step,
            'step_normalized': step_normalized,
            'unique_tokens_ratio': unique_tokens_ratio,
            'repetition_ratio': repetition_ratio,
            'sequence_score': sequence_score,
            'sequence_reward': sequence_score,  # Use same value for simplicity
            'prompt_length_squared': prompt_length_squared,
            'prompt_length_log': prompt_length_log,
            'prompt_step_interaction': prompt_step_interaction
        }
        
    def train_from_data(self, data_dir="/root/code/verl/outputs"):
        """
        Train the model from VERL training data.
        """
        print("Training response length predictor...")
        
        # Load and prepare data (simplified version)
        from response_length_analysis import ResponseLengthAnalyzer
        
        analyzer = ResponseLengthAnalyzer(data_dir)
        analyzer.load_training_data()
        
        if analyzer.training_data is None:
            raise ValueError("No training data found")
            
        # Extract features
        features_df = analyzer.extract_prompt_features(analyzer.training_data)
        X, y = analyzer.prepare_features_for_modeling(features_df)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        
        print(f"Model trained successfully!")
        print(f"Training RMSE: {rmse:.2f}")
        print(f"Training MAE: {mae:.2f}")
        print(f"Training R²: {r2:.3f}")
        
        return self
        
    def predict_response_length(self, prompt_length, training_step=350, 
                              sequence_score=0.7, return_confidence=False):
        """
        Predict response length for given prompt characteristics.
        
        Args:
            prompt_length (int): Length of prompt in tokens
            training_step (int): Training step (default: 350 for final model)
            sequence_score (float): Expected quality score 0-1
            return_confidence (bool): Whether to return prediction confidence
            
        Returns:
            int or tuple: Predicted response length (and confidence if requested)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Calculate features
        features = self.calculate_prompt_features(
            prompt_length, training_step, sequence_score
        )
        
        # Convert to DataFrame with correct feature order
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[self.feature_names]
        
        # Make prediction
        prediction = self.model.predict(feature_df)[0]
        
        # Clip to reasonable bounds
        prediction = max(10, min(2048, prediction))
        
        if return_confidence:
            # Estimate confidence based on input similarity to training data
            confidence = self._estimate_confidence(features)
            return int(prediction), confidence
        
        return int(prediction)
        
    def _estimate_confidence(self, features):
        """
        Estimate prediction confidence based on feature values.
        """
        # Simple heuristic: confidence is higher for typical values
        prompt_len = features['prompt_length']
        
        # Confidence decreases for extreme prompt lengths
        if prompt_len < 60 or prompt_len > 200:
            confidence = 0.6
        else:
            confidence = 0.8
            
        # Adjust based on training step
        if features['training_step'] > 300:
            confidence += 0.1
        elif features['training_step'] < 50:
            confidence -= 0.1
            
        return max(0.3, min(0.95, confidence))
        
    def get_length_recommendations(self, target_length, training_step=350):
        """
        Get prompt length recommendations to achieve target response length.
        
        Args:
            target_length (int): Desired response length
            training_step (int): Training step context
            
        Returns:
            dict: Recommendations for achieving target length
        """
        recommendations = []
        
        # Test different prompt lengths
        test_lengths = range(60, 250, 10)
        
        for prompt_len in test_lengths:
            pred_length = self.predict_response_length(prompt_len, training_step)
            error = abs(pred_length - target_length)
            recommendations.append({
                'prompt_length': prompt_len,
                'predicted_response': pred_length,
                'error': error
            })
        
        # Sort by error
        recommendations.sort(key=lambda x: x['error'])
        
        best_rec = recommendations[0]
        
        return {
            'target_response_length': target_length,
            'recommended_prompt_length': best_rec['prompt_length'],
            'predicted_response_length': best_rec['predicted_response'],
            'prediction_error': best_rec['error'],
            'alternative_options': recommendations[1:6]  # Top 5 alternatives
        }
        
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")
        
    def analyze_prompt_impact(self, base_prompt_length=100, training_step=350):
        """
        Analyze how different factors impact response length.
        
        Returns:
            dict: Analysis of various factors
        """
        base_prediction = self.predict_response_length(base_prompt_length, training_step)
        
        analysis = {
            'base_case': {
                'prompt_length': base_prompt_length,
                'training_step': training_step,
                'predicted_response': base_prediction
            },
            'prompt_length_impact': {},
            'training_step_impact': {},
            'quality_impact': {}
        }
        
        # Prompt length impact
        for length in [50, 75, 100, 125, 150, 200]:
            pred = self.predict_response_length(length, training_step)
            analysis['prompt_length_impact'][length] = {
                'predicted_response': pred,
                'change_from_base': pred - base_prediction
            }
            
        # Training step impact
        for step in [50, 150, 250, 350]:
            pred = self.predict_response_length(base_prompt_length, step)
            analysis['training_step_impact'][step] = {
                'predicted_response': pred,
                'change_from_base': pred - base_prediction
            }
            
        # Quality score impact
        for score in [0.2, 0.5, 0.7, 0.9]:
            pred = self.predict_response_length(base_prompt_length, training_step, score)
            analysis['quality_impact'][score] = {
                'predicted_response': pred,
                'change_from_base': pred - base_prediction
            }
            
        return analysis

def demo_predictor():
    """Demonstrate the response length predictor."""
    print("=== Response Length Predictor Demo ===\n")
    
    # Create and train predictor
    predictor = ResponseLengthPredictor()
    
    # Check if we can train from data
    try:
        predictor.train_from_data()
    except Exception as e:
        print(f"Could not train from data: {e}")
        print("Creating a demo predictor with simulated model...")
        
        # Create a simple demo model for illustration
        from sklearn.ensemble import GradientBoostingRegressor
        predictor.model = GradientBoostingRegressor(random_state=42)
        
        # Create demo training data
        np.random.seed(42)
        n_samples = 1000
        demo_features = []
        demo_targets = []
        
        for _ in range(n_samples):
            prompt_len = np.random.randint(60, 200)
            step = np.random.randint(1, 350)
            score = np.random.uniform(0.1, 0.9)
            
            features = predictor.calculate_prompt_features(prompt_len, step, score)
            feature_values = [features[name] for name in predictor.feature_names]
            
            # Simulate target (simple relationship)
            target = prompt_len * 2.5 + np.random.normal(0, 50)
            target = max(50, min(800, target))
            
            demo_features.append(feature_values)
            demo_targets.append(target)
        
        predictor.model.fit(demo_features, demo_targets)
        predictor.is_trained = True
        print("Demo model created and trained with simulated data.\n")
    
    # Demo predictions
    print("1. Basic Predictions:")
    for prompt_len in [75, 100, 150, 200]:
        prediction = predictor.predict_response_length(prompt_len)
        print(f"   Prompt length {prompt_len} → Predicted response: {prediction} tokens")
    
    print("\n2. Predictions with Confidence:")
    prediction, confidence = predictor.predict_response_length(120, return_confidence=True)
    print(f"   Prompt length 120 → Response: {prediction} tokens (confidence: {confidence:.2f})")
    
    print("\n3. Target Length Recommendations:")
    recommendations = predictor.get_length_recommendations(target_length=300)
    print(f"   To get ~300 token response:")
    print(f"   → Use prompt length: {recommendations['recommended_prompt_length']} tokens")
    print(f"   → Expected response: {recommendations['predicted_response_length']} tokens")
    
    print("\n4. Factor Impact Analysis:")
    analysis = predictor.analyze_prompt_impact()
    print("   Prompt length impact on response length:")
    for length, impact in analysis['prompt_length_impact'].items():
        change = impact['change_from_base']
        print(f"     {length} tokens: {impact['predicted_response']} response ({change:+.0f} vs base)")
    
    # Save model for later use
    predictor.save_model('/root/code/verl/response_length_predictor.pkl')
    print(f"\n5. Model saved to response_length_predictor.pkl")
    
    return predictor

if __name__ == "__main__":
    predictor = demo_predictor()