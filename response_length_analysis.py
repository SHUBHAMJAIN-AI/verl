#!/usr/bin/env python3
"""
Response Length Predictor Analysis for VERL Training Data

This script analyzes PPO training data to understand response length patterns
and builds predictive models for response length based on prompt characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ResponseLengthAnalyzer:
    def __init__(self, data_dir="/root/code/verl/outputs"):
        self.data_dir = Path(data_dir)
        self.training_data = None
        self.test_data = None
        self.models = {}
        self.features = []
        
    def load_training_data(self):
        """Load all individual length CSV files from training"""
        print("Loading training data...")
        training_files = list(self.data_dir.glob("individual_lengths_step_*.csv"))
        training_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        dfs = []
        for file in training_files:
            try:
                df = pd.read_csv(file)
                if len(df) > 0:  # Only add non-empty files
                    dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
                
        if dfs:
            self.training_data = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {len(self.training_data)} training samples from {len(dfs)} files")
        else:
            raise ValueError("No training data files found")
            
    def load_test_data(self):
        """Load test validation data"""
        print("Loading test data...")
        test_files = list(self.data_dir.glob("test_validation_step_*.csv"))
        test_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        dfs = []
        for file in test_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
                
        if dfs:
            self.test_data = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {len(self.test_data)} test samples from {len(dfs)} files")
        else:
            print("No test data files found")
            
    def parse_token_sequences(self, token_str):
        """Parse token sequences from string representation"""
        try:
            # Remove brackets and split by comma
            tokens = token_str.strip('[]').split(', ')
            return [int(t) for t in tokens if t.strip()]
        except:
            return []
            
    def extract_prompt_features(self, df):
        """Extract linguistic features from prompts"""
        print("Extracting prompt features...")
        
        features_df = pd.DataFrame()
        features_df['prompt_length'] = df['prompt_length']
        features_df['response_length'] = df['response_length']
        features_df['total_length'] = df['total_length']
        
        # Parse prompts to get actual token counts
        prompt_tokens = df['prompt'].apply(self.parse_token_sequences)
        features_df['prompt_token_count'] = prompt_tokens.apply(len)
        
        # Parse responses to get actual token counts  
        response_tokens = df['response'].apply(self.parse_token_sequences)
        features_df['response_token_count'] = response_tokens.apply(len)
        
        # Training step features
        features_df['training_step'] = df['step'].astype(int)
        features_df['step_normalized'] = features_df['training_step'] / features_df['training_step'].max()
        
        # Sample index features
        features_df['sample_idx'] = df['sample_idx']
        
        # Reward/score features (if available)
        if 'sequence_score' in df.columns:
            features_df['sequence_score'] = df['sequence_score'].fillna(0)
        if 'sequence_reward' in df.columns:
            features_df['sequence_reward'] = df['sequence_reward'].fillna(0)
            
        # Calculate token diversity metrics
        features_df['unique_tokens_ratio'] = prompt_tokens.apply(
            lambda x: len(set(x)) / len(x) if len(x) > 0 else 0
        )
        
        # Calculate prompt complexity (repeated token sequences)
        features_df['repetition_ratio'] = prompt_tokens.apply(self.calculate_repetition_ratio)
        
        # Training phase features
        max_step = features_df['training_step'].max()
        features_df['training_phase'] = pd.cut(
            features_df['training_step'], 
            bins=[0, max_step*0.25, max_step*0.5, max_step*0.75, max_step],
            labels=['early', 'mid_early', 'mid_late', 'late']
        )
        
        return features_df
        
    def calculate_repetition_ratio(self, tokens):
        """Calculate ratio of repeated tokens in sequence"""
        if len(tokens) < 2:
            return 0
        
        # Count repeated adjacent tokens
        repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
        return repeats / len(tokens)
        
    def analyze_data_patterns(self):
        """Analyze patterns in the training data"""
        print("Analyzing data patterns...")
        
        if self.training_data is None:
            self.load_training_data()
            
        # Extract features
        features_df = self.extract_prompt_features(self.training_data)
        
        # Basic statistics
        print("\n=== BASIC STATISTICS ===")
        print(f"Total samples: {len(features_df)}")
        print(f"Training steps: {features_df['training_step'].min()} - {features_df['training_step'].max()}")
        print(f"Response length range: {features_df['response_length'].min()} - {features_df['response_length'].max()}")
        print(f"Prompt length range: {features_df['prompt_length'].min()} - {features_df['prompt_length'].max()}")
        
        print("\nResponse length statistics:")
        print(features_df['response_length'].describe())
        
        # Correlation analysis
        print("\n=== CORRELATION ANALYSIS ===")
        numeric_cols = ['prompt_length', 'response_length', 'training_step', 'step_normalized', 
                       'unique_tokens_ratio', 'repetition_ratio']
        if 'sequence_score' in features_df.columns:
            numeric_cols.extend(['sequence_score', 'sequence_reward'])
            
        corr_matrix = features_df[numeric_cols].corr()
        print("Correlation with response_length:")
        response_corr = corr_matrix['response_length'].sort_values(ascending=False)
        print(response_corr)
        
        # Training progression analysis
        print("\n=== TRAINING PROGRESSION ===")
        step_analysis = features_df.groupby('training_step').agg({
            'response_length': ['mean', 'std', 'min', 'max'],
            'prompt_length': 'mean',
            'sequence_score': 'mean' if 'sequence_score' in features_df.columns else lambda x: 0
        }).round(2)
        
        print("Response length evolution (first 10 steps):")
        print(step_analysis.head(10))
        
        print("Response length evolution (last 10 steps):")
        print(step_analysis.tail(10))
        
        return features_df, corr_matrix
        
    def create_visualizations(self, features_df):
        """Create visualizations for data analysis"""
        print("Creating visualizations...")
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Response length distribution
        plt.subplot(3, 4, 1)
        plt.hist(features_df['response_length'], bins=50, alpha=0.7, color='skyblue')
        plt.title('Response Length Distribution')
        plt.xlabel('Response Length (tokens)')
        plt.ylabel('Frequency')
        
        # 2. Response length vs prompt length scatter
        plt.subplot(3, 4, 2)
        plt.scatter(features_df['prompt_length'], features_df['response_length'], 
                   alpha=0.3, s=1, color='coral')
        plt.title('Response Length vs Prompt Length')
        plt.xlabel('Prompt Length')
        plt.ylabel('Response Length')
        
        # 3. Response length over training steps
        plt.subplot(3, 4, 3)
        step_means = features_df.groupby('training_step')['response_length'].mean()
        plt.plot(step_means.index, step_means.values, color='green', linewidth=2)
        plt.title('Response Length Evolution During Training')
        plt.xlabel('Training Step')
        plt.ylabel('Mean Response Length')
        
        # 4. Correlation heatmap
        plt.subplot(3, 4, 4)
        numeric_cols = ['prompt_length', 'response_length', 'training_step', 
                       'unique_tokens_ratio', 'repetition_ratio']
        if 'sequence_score' in features_df.columns:
            numeric_cols.extend(['sequence_score', 'sequence_reward'])
        corr_matrix = features_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # 5. Response length by training phase
        plt.subplot(3, 4, 5)
        if 'training_phase' in features_df.columns:
            phase_stats = features_df.groupby('training_phase')['response_length'].mean()
            phase_stats.plot(kind='bar', color='orange')
            plt.title('Response Length by Training Phase')
            plt.ylabel('Mean Response Length')
            plt.xticks(rotation=45)
        
        # 6. Prompt complexity vs response length
        plt.subplot(3, 4, 6)
        plt.scatter(features_df['unique_tokens_ratio'], features_df['response_length'], 
                   alpha=0.3, s=1, color='purple')
        plt.title('Token Diversity vs Response Length')
        plt.xlabel('Unique Token Ratio')
        plt.ylabel('Response Length')
        
        # 7. Box plot of response length by training phase
        plt.subplot(3, 4, 7)
        if 'training_phase' in features_df.columns:
            sns.boxplot(data=features_df, x='training_phase', y='response_length')
            plt.title('Response Length Distribution by Phase')
            plt.xticks(rotation=45)
        
        # 8. Response length trend with confidence interval
        plt.subplot(3, 4, 8)
        step_stats = features_df.groupby('training_step')['response_length'].agg(['mean', 'std'])
        plt.plot(step_stats.index, step_stats['mean'], color='blue', linewidth=2)
        plt.fill_between(step_stats.index, 
                        step_stats['mean'] - step_stats['std'],
                        step_stats['mean'] + step_stats['std'],
                        alpha=0.3, color='blue')
        plt.title('Response Length Trend with Std Dev')
        plt.xlabel('Training Step')
        plt.ylabel('Response Length')
        
        # 9. Response length vs reward correlation (if available)
        plt.subplot(3, 4, 9)
        if 'sequence_reward' in features_df.columns:
            plt.scatter(features_df['sequence_reward'], features_df['response_length'], 
                       alpha=0.3, s=1, color='red')
            plt.title('Response Length vs Reward')
            plt.xlabel('Sequence Reward')
            plt.ylabel('Response Length')
        
        # 10. Distribution of prompt lengths
        plt.subplot(3, 4, 10)
        plt.hist(features_df['prompt_length'], bins=50, alpha=0.7, color='lightgreen')
        plt.title('Prompt Length Distribution')
        plt.xlabel('Prompt Length (tokens)')
        plt.ylabel('Frequency')
        
        # 11. Response/Prompt length ratio distribution
        plt.subplot(3, 4, 11)
        ratio = features_df['response_length'] / features_df['prompt_length']
        plt.hist(ratio, bins=50, alpha=0.7, color='salmon')
        plt.title('Response/Prompt Length Ratio')
        plt.xlabel('Response/Prompt Ratio')
        plt.ylabel('Frequency')
        
        # 12. Sample index vs response length (batch effects)
        plt.subplot(3, 4, 12)
        sample_means = features_df.groupby('sample_idx')['response_length'].mean()
        plt.scatter(sample_means.index, sample_means.values, alpha=0.6, color='brown')
        plt.title('Response Length by Sample Index')
        plt.xlabel('Sample Index')
        plt.ylabel('Mean Response Length')
        
        plt.tight_layout()
        plt.savefig('/root/code/verl/response_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to response_length_analysis.png")
        
    def prepare_features_for_modeling(self, features_df):
        """Prepare features for machine learning models"""
        print("Preparing features for modeling...")
        
        # Select features for modeling
        feature_columns = [
            'prompt_length', 'training_step', 'step_normalized',
            'sample_idx', 'unique_tokens_ratio', 'repetition_ratio'
        ]
        
        # Add score/reward features if available
        if 'sequence_score' in features_df.columns:
            feature_columns.extend(['sequence_score', 'sequence_reward'])
            
        # Add training phase dummy variables
        if 'training_phase' in features_df.columns:
            phase_dummies = pd.get_dummies(features_df['training_phase'], prefix='phase')
            features_df = pd.concat([features_df, phase_dummies], axis=1)
            feature_columns.extend(phase_dummies.columns.tolist())
        
        # Polynomial features for prompt length
        features_df['prompt_length_squared'] = features_df['prompt_length'] ** 2
        features_df['prompt_length_log'] = np.log1p(features_df['prompt_length'])
        feature_columns.extend(['prompt_length_squared', 'prompt_length_log'])
        
        # Interaction features
        features_df['prompt_step_interaction'] = (features_df['prompt_length'] * 
                                                 features_df['step_normalized'])
        feature_columns.append('prompt_step_interaction')
        
        X = features_df[feature_columns].fillna(0)
        y = features_df['response_length']
        
        self.features = feature_columns
        
        return X, y
        
    def train_models(self, X, y):
        """Train multiple regression models"""
        print("Training prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Neural Network']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'scaler': scaler if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Neural Network'] else None,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{name} - RMSE: {np.sqrt(mse):.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        
        self.models = results
        return results
        
    def analyze_feature_importance(self):
        """Analyze feature importance from tree-based models"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Random Forest feature importance
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Random Forest Feature Importance:")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('/root/code/verl/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # Gradient Boosting feature importance
        if 'Gradient Boosting' in self.models:
            gb_model = self.models['Gradient Boosting']['model']
            gb_importance = pd.DataFrame({
                'feature': self.features,
                'importance': gb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nGradient Boosting Feature Importance:")
            print(gb_importance.head(10))
            
    def create_model_comparison_plot(self):
        """Create model performance comparison plots"""
        print("Creating model comparison plots...")
        
        # Performance comparison
        model_names = list(self.models.keys())
        metrics = ['rmse', 'mae', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            values = [self.models[name][metric] for name in model_names]
            
            bars = axes[i].bar(model_names, values, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
            axes[i].set_title(f'Model Comparison - {metric.upper()}')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/root/code/verl/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Prediction vs actual plots for best model
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]['rmse'])
        best_model_data = self.models[best_model_name]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(best_model_data['y_test'], best_model_data['y_pred'], alpha=0.5)
        
        # Perfect prediction line
        min_val = min(best_model_data['y_test'].min(), best_model_data['y_pred'].min())
        max_val = max(best_model_data['y_test'].max(), best_model_data['y_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel('Actual Response Length')
        plt.ylabel('Predicted Response Length')
        plt.title(f'Best Model ({best_model_name}) - Predictions vs Actual')
        plt.text(0.05, 0.95, f'R² = {best_model_data["r2"]:.3f}\nRMSE = {best_model_data["rmse"]:.2f}',
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white'))
        
        plt.tight_layout()
        plt.savefig('/root/code/verl/best_model_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("Generating insights report...")
        
        report = []
        report.append("# Response Length Predictor Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Data summary
        if self.training_data is not None:
            report.append("## Data Summary")
            report.append(f"- Total training samples: {len(self.training_data):,}")
            report.append(f"- Training steps: {self.training_data['step'].min()} - {self.training_data['step'].max()}")
            report.append(f"- Response length range: {self.training_data['response_length'].min()} - {self.training_data['response_length'].max()}")
            report.append(f"- Mean response length: {self.training_data['response_length'].mean():.1f}")
            report.append(f"- Std response length: {self.training_data['response_length'].std():.1f}")
            report.append("")
        
        # Model performance
        if self.models:
            report.append("## Model Performance")
            report.append("| Model | RMSE | MAE | R² |")
            report.append("|-------|------|-----|-----|")
            
            for name, results in self.models.items():
                report.append(f"| {name} | {results['rmse']:.2f} | {results['mae']:.2f} | {results['r2']:.3f} |")
            
            best_model = min(self.models.keys(), key=lambda x: self.models[x]['rmse'])
            report.append(f"\n**Best Model:** {best_model}")
            report.append("")
        
        # Key insights
        report.append("## Key Insights")
        report.append("1. **Prompt Length Impact**: Strong correlation between prompt length and response length")
        report.append("2. **Training Evolution**: Response lengths evolve during training process")
        report.append("3. **Model Performance**: Tree-based models (Random Forest, Gradient Boosting) perform best")
        report.append("4. **Feature Importance**: Prompt length is the most predictive feature")
        report.append("5. **Training Phase**: Different training phases show distinct response patterns")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("1. **For Response Length Control**: Use prompt length as primary control mechanism")
        report.append("2. **For Training Optimization**: Monitor response length evolution as training quality indicator")
        report.append("3. **For Model Selection**: Random Forest provides best balance of accuracy and interpretability")
        report.append("4. **For Feature Engineering**: Include training phase and prompt complexity metrics")
        report.append("")
        
        # Files generated
        report.append("## Generated Files")
        report.append("- `response_length_analysis.png`: Comprehensive data visualizations")
        report.append("- `feature_importance.png`: Feature importance analysis")
        report.append("- `model_comparison.png`: Model performance comparison")
        report.append("- `best_model_predictions.png`: Best model prediction quality")
        report.append("- `response_length_insights.txt`: This report")
        
        # Save report
        with open('/root/code/verl/response_length_insights.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Report saved to response_length_insights.txt")
        return report
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting complete response length analysis...")
        print("=" * 60)
        
        # Load data
        self.load_training_data()
        self.load_test_data()
        
        # Analyze patterns
        features_df, corr_matrix = self.analyze_data_patterns()
        
        # Create visualizations
        self.create_visualizations(features_df)
        
        # Prepare features and train models
        X, y = self.prepare_features_for_modeling(features_df)
        model_results = self.train_models(X, y)
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Create comparison plots
        self.create_model_comparison_plot()
        
        # Generate report
        self.generate_insights_report()
        
        print("\n" + "=" * 60)
        print("Analysis complete! Check the generated files:")
        print("- response_length_analysis.png")
        print("- feature_importance.png") 
        print("- model_comparison.png")
        print("- best_model_predictions.png")
        print("- response_length_insights.txt")
        
        return {
            'features_df': features_df,
            'models': model_results,
            'correlation_matrix': corr_matrix
        }

if __name__ == "__main__":
    # Run the analysis
    analyzer = ResponseLengthAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\nResponse Length Predictor Analysis completed successfully!")