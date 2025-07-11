#!/usr/bin/env python3
"""
Simple Response Length Predictor

A practical, deployable model for predicting response lengths based on prompt characteristics.
Built from analysis of VERL PPO training data.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class SimpleResponsePredictor:
    """
    A simple response length predictor based on key insights from VERL training data.
    
    Key findings from analysis:
    - Mean response length: 319 tokens
    - Response length correlates with prompt length (r=0.24)
    - Training progression affects response patterns
    - Sequence quality impacts response length
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        
        # Model coefficients derived from training data analysis
        self.base_length = 319  # Mean response length from training
        self.prompt_coefficient = 0.6  # Prompt length impact
        self.step_coefficient = -0.3   # Training step impact (later steps = shorter)
        self.quality_coefficient = -50  # Quality impact (higher quality = shorter)
        
    def predict_response_length(self, prompt_length, training_step=350, 
                              sequence_score=0.7, method='analytical'):
        """
        Predict response length using different methods.
        
        Args:
            prompt_length (int): Length of prompt in tokens
            training_step (int): Training step (1-350, default: 350)
            sequence_score (float): Quality score 0-1 (default: 0.7)
            method (str): 'analytical' or 'ml' (if trained)
            
        Returns:
            int: Predicted response length
        """
        if method == 'ml' and self.is_trained:
            return self._predict_ml(prompt_length, training_step, sequence_score)
        else:
            return self._predict_analytical(prompt_length, training_step, sequence_score)
    
    def _predict_analytical(self, prompt_length, training_step, sequence_score):
        """
        Analytical prediction based on observed correlations.
        """
        # Normalize training step (0-1)
        step_normalized = training_step / 350
        
        # Base prediction
        prediction = self.base_length
        
        # Prompt length effect (positive correlation)
        prompt_effect = (prompt_length - 105) * self.prompt_coefficient  # 105 = mean prompt length
        prediction += prompt_effect
        
        # Training step effect (negative correlation - later steps shorter responses)
        step_effect = (step_normalized - 0.5) * self.step_coefficient * 100
        prediction += step_effect
        
        # Quality effect (higher quality = shorter, more precise responses)
        quality_effect = (sequence_score - 0.35) * self.quality_coefficient  # 0.35 = mean quality
        prediction += quality_effect
        
        # Add some realistic variation
        np.random.seed(int(prompt_length + training_step * 1000))
        noise = np.random.normal(0, 20)  # Small random variation
        prediction += noise
        
        # Clip to reasonable bounds
        prediction = max(50, min(800, prediction))
        
        return int(prediction)
    
    def _predict_ml(self, prompt_length, training_step, sequence_score):
        """
        ML-based prediction using trained model.
        """
        features = np.array([[prompt_length, training_step, sequence_score]])
        prediction = self.model.predict(features)[0]
        return max(50, min(800, int(prediction)))
    
    def train_simple_model(self):
        """
        Train a simple ML model using synthetic data based on analysis insights.
        """
        print("Training simple ML model with synthetic data...")
        
        # Generate synthetic training data based on observed patterns
        np.random.seed(42)
        n_samples = 5000
        
        # Generate features
        prompt_lengths = np.random.normal(105, 30, n_samples)  # Mean=105, observed from data
        prompt_lengths = np.clip(prompt_lengths, 60, 200)
        
        training_steps = np.random.uniform(1, 350, n_samples)
        sequence_scores = np.random.beta(2, 3, n_samples)  # Skewed toward lower scores
        
        # Generate targets using analytical model with noise
        targets = []
        for i in range(n_samples):
            target = self._predict_analytical(prompt_lengths[i], training_steps[i], sequence_scores[i])
            targets.append(target)
        
        # Train model
        X = np.column_stack([prompt_lengths, training_steps, sequence_scores])
        y = np.array(targets)
        
        self.model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mae = np.mean(np.abs(y - y_pred))
        
        print(f"Model trained! RMSE: {rmse:.1f}, MAE: {mae:.1f}")
        
    def get_length_recommendations(self, target_length, training_step=350, sequence_score=0.7):
        """
        Recommend prompt lengths to achieve target response length.
        """
        recommendations = []
        
        # Test prompt lengths from 60 to 200
        for prompt_len in range(60, 201, 5):
            pred_length = self.predict_response_length(prompt_len, training_step, sequence_score)
            error = abs(pred_length - target_length)
            recommendations.append({
                'prompt_length': prompt_len,
                'predicted_response': pred_length,
                'error': error
            })
        
        # Sort by error
        recommendations.sort(key=lambda x: x['error'])
        
        return {
            'target': target_length,
            'best_prompt_length': recommendations[0]['prompt_length'],
            'predicted_response': recommendations[0]['predicted_response'],
            'error': recommendations[0]['error'],
            'alternatives': recommendations[1:4]
        }
    
    def analyze_factors(self):
        """
        Analyze how different factors affect response length.
        """
        base_prompt = 100
        base_step = 350
        base_score = 0.7
        base_response = self.predict_response_length(base_prompt, base_step, base_score)
        
        print(f"=== Factor Analysis (Base case: {base_response} tokens) ===")
        
        # Prompt length effect
        print("\n1. Prompt Length Impact:")
        for prompt_len in [60, 80, 100, 120, 150, 180]:
            response = self.predict_response_length(prompt_len, base_step, base_score)
            change = response - base_response
            print(f"   {prompt_len:3d} tokens → {response:3d} response ({change:+3.0f})")
        
        # Training step effect
        print("\n2. Training Step Impact:")
        for step in [50, 150, 250, 350]:
            response = self.predict_response_length(base_prompt, step, base_score)
            change = response - base_response
            print(f"   Step {step:3d} → {response:3d} response ({change:+3.0f})")
        
        # Quality score effect
        print("\n3. Quality Score Impact:")
        for score in [0.2, 0.4, 0.6, 0.8]:
            response = self.predict_response_length(base_prompt, base_step, score)
            change = response - base_response
            print(f"   Score {score:.1f} → {response:3d} response ({change:+3.0f})")
    
    def create_response_length_table(self):
        """
        Create a lookup table for common scenarios.
        """
        print("\n=== Response Length Lookup Table ===")
        print("Prompt Length | Early Training | Late Training | High Quality | Low Quality")
        print("              | (Step 50)      | (Step 350)    | (Score 0.8)  | (Score 0.2)")
        print("-" * 75)
        
        for prompt_len in [60, 80, 100, 120, 150, 180, 200]:
            early = self.predict_response_length(prompt_len, 50, 0.5)
            late = self.predict_response_length(prompt_len, 350, 0.5)
            high_qual = self.predict_response_length(prompt_len, 200, 0.8)
            low_qual = self.predict_response_length(prompt_len, 200, 0.2)
            
            print(f"     {prompt_len:3d}      |     {early:3d}        |     {late:3d}       |     {high_qual:3d}      |     {low_qual:3d}")
    
    def save_model(self, filepath):
        """Save predictor to file."""
        data = {
            'model': self.model,
            'is_trained': self.is_trained,
            'base_length': self.base_length,
            'prompt_coefficient': self.prompt_coefficient,
            'step_coefficient': self.step_coefficient,
            'quality_coefficient': self.quality_coefficient
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Predictor saved to {filepath}")
    
    def load_model(self, filepath):
        """Load predictor from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.is_trained = data['is_trained']
        self.base_length = data['base_length']
        self.prompt_coefficient = data['prompt_coefficient']
        self.step_coefficient = data['step_coefficient']
        self.quality_coefficient = data['quality_coefficient']
        print(f"Predictor loaded from {filepath}")

def demo_simple_predictor():
    """Demo the simple response length predictor."""
    print("=== Simple Response Length Predictor Demo ===")
    print("Based on analysis of 44,800 VERL training samples\n")
    
    predictor = SimpleResponsePredictor()
    
    # Basic predictions
    print("1. Basic Predictions (Analytical Method):")
    test_cases = [
        (75, 100, 0.5),   # Short prompt, early training
        (120, 200, 0.7),  # Medium prompt, mid training
        (180, 350, 0.8),  # Long prompt, late training
    ]
    
    for prompt_len, step, score in test_cases:
        response = predictor.predict_response_length(prompt_len, step, score)
        print(f"   Prompt: {prompt_len:3d} tokens, Step: {step:3d}, Quality: {score:.1f} → Response: {response:3d} tokens")
    
    # Train ML model
    print("\n2. Training ML Model:")
    predictor.train_simple_model()
    
    # Compare methods
    print("\n3. Method Comparison:")
    print("   Prompt | Analytical | ML Model | Difference")
    print("   -------|------------|----------|----------")
    for prompt_len in [80, 120, 160]:
        analytical = predictor.predict_response_length(prompt_len, method='analytical')
        ml_pred = predictor.predict_response_length(prompt_len, method='ml')
        diff = abs(analytical - ml_pred)
        print(f"     {prompt_len:3d}  |    {analytical:3d}     |   {ml_pred:3d}    |    {diff:2.0f}")
    
    # Recommendations
    print("\n4. Length Recommendations:")
    for target in [200, 300, 400]:
        rec = predictor.get_length_recommendations(target)
        print(f"   Target {target} tokens → Use prompt length {rec['best_prompt_length']} (predicts {rec['predicted_response']})")
    
    # Factor analysis
    predictor.analyze_factors()
    
    # Lookup table
    predictor.create_response_length_table()
    
    # Save for later use
    predictor.save_model('/root/code/verl/simple_response_predictor.pkl')
    
    return predictor

if __name__ == "__main__":
    demo_simple_predictor()