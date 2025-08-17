# BigMart Sales Prediction - Model Experimentation Notebook

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

class BigMartModelExperiments:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
    
    def load_processed_data(self):
        """Load preprocessed data"""
        print("Loading processed data...")
        
    def prepare_data_for_experiments(self, X, y):
        """Prepare data for model experiments"""
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        print(f"Training set size: {self.X_train.shape}")
        print(f"Validation set size: {self.X_val.shape}")
    
    def define_models(self):
        """Define all models to experiment with"""
        self.models = {
            # Linear Models
            'Linear_Regression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            
            # Tree-based Models
            'Decision_Tree': DecisionTreeRegressor(random_state=42),
            'Random_Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Extra_Trees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            
            # Gradient Boosting Models
            'Gradient_Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=False),
            
            # Other Models
            'KNN': KNeighborsRegressor(n_jobs=-1),
            'SVR': SVR()
        }
        
        print(f"Defined {len(self.models)} models for experimentation")
    
    def evaluate_model(self, name, model, X_train, X_val, y_train, y_val):
        """Evaluate a single model"""
        try:
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-cv_scores.mean())
            cv_std = np.sqrt(cv_scores.std())
            
            results = {
                'Model': name,
                'Train_RMSE': train_rmse,
                'Val_RMSE': val_rmse,
                'Val_MAE': val_mae,
                'Val_R2': val_r2,
                'CV_RMSE': cv_rmse,
                'CV_Std': cv_std,
                'Overfit_Ratio': val_rmse / train_rmse,
                'Model_Object': model
            }
            
            return results
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            return None
    
    def run_baseline_experiments(self):
        """Run baseline experiments with all models"""
        print("\n" + "="*80)
        print("BASELINE MODEL EXPERIMENTS")
        print("="*80)
        
        self.define_models()
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            result = self.evaluate_model(name, model, self.X_train, self.X_val, 
                                       self.y_train, self.y_val)
            if result:
                self.results[name] = result
        
        # Create results dataframe
        results_df = pd.DataFrame([
            {k: v for k, v in result.items() if k != 'Model_Object'} 
            for result in self.results.values()
        ])
        
        # Sort by validation RMSE
        results_df = results_df.sort_values('Val_RMSE')
        
        print("\n" + "="*80)
        print("BASELINE RESULTS SUMMARY")
        print("="*80)
        print(results_df.round(4))
        
        # Identify best model
        best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.results[best_model_name]['Model_Object']
        print(f"\nBest baseline model: {best_model_name}")
        print(f"Best Val RMSE: {results_df.iloc[0]['Val_RMSE']:.4f}")
        
        return results_df
    
    def hyperparameter_tuning_xgboost(self):
        """Advanced hyperparameter tuning for XGBoost"""
        print("\n" + "="*60)
        print("XGBOOST HYPERPARAMETER TUNING")
        print("="*60)
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 5, 10]
        }
        
        # Randomized search for efficiency
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=50, cv=5,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        # Best model evaluation
        best_xgb = random_search.best_estimator_
        y_pred = best_xgb.predict(self.X_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        
        print(f"Best XGBoost parameters: {random_search.best_params_}")
        print(f"Best XGBoost Val RMSE: {rmse:.4f}")
        
        return best_xgb, rmse
    
    def hyperparameter_tuning_lightgbm(self):
        """Advanced hyperparameter tuning for LightGBM"""
        print("\n" + "="*60)
        print("LIGHTGBM HYPERPARAMETER TUNING")
        print("="*60)
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 5, 10],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [10, 20, 30]
        }
        
        # Randomized search
        lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        random_search = RandomizedSearchCV(
            lgb_model, param_grid, n_iter=50, cv=5,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        # Best model evaluation
        best_lgb = random_search.best_estimator_
        y_pred = best_lgb.predict(self.X_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        
        print(f"Best LightGBM parameters: {random_search.best_params_}")
        print(f"Best LightGBM Val RMSE: {rmse:.4f}")
        
        return best_lgb, rmse
    
    def hyperparameter_tuning_random_forest(self):
        """Advanced hyperparameter tuning for Random Forest"""
        print("\n" + "="*60)
        print("RANDOM FOREST HYPERPARAMETER TUNING")
        print("="*60)
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Randomized search
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(
            rf_model, param_grid, n_iter=50, cv=5,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        # Best model evaluation
        best_rf = random_search.best_estimator_
        y_pred = best_rf.predict(self.X_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        
        print(f"Best Random Forest parameters: {random_search.best_params_}")
        print(f"Best Random Forest Val RMSE: {rmse:.4f}")
        
        return best_rf, rmse
    
    def ensemble_models(self):
        """Create ensemble models"""
        print("\n" + "="*60)
        print("ENSEMBLE MODELING")
        print("="*60)
        
        # Get best models from tuning
        best_xgb, xgb_rmse = self.hyperparameter_tuning_xgboost()
        best_lgb, lgb_rmse = self.hyperparameter_tuning_lightgbm()
        best_rf, rf_rmse = self.hyperparameter_tuning_random_forest()
        
        # Simple ensemble (average)
        xgb_pred = best_xgb.predict(self.X_val)
        lgb_pred = best_lgb.predict(self.X_val)
        rf_pred = best_rf.predict(self.X_val)
        
        # Weighted ensemble based on performance
        total_weight = 1/xgb_rmse + 1/lgb_rmse + 1/rf_rmse
        xgb_weight = (1/xgb_rmse) / total_weight
        lgb_weight = (1/lgb_rmse) / total_weight
        rf_weight = (1/rf_rmse) / total_weight
        
        ensemble_pred = (xgb_weight * xgb_pred + 
                        lgb_weight * lgb_pred + 
                        rf_weight * rf_pred)
        
        ensemble_rmse = np.sqrt(mean_squared_error(self.y_val, ensemble_pred))
        
        print(f"Ensemble weights - XGB: {xgb_weight:.3f}, LGB: {lgb_weight:.3f}, RF: {rf_weight:.3f}")
        print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
        
        # Compare with individual models
        print(f"XGBoost RMSE: {xgb_rmse:.4f}")
        print(f"LightGBM RMSE: {lgb_rmse:.4f}")
        print(f"Random Forest RMSE: {rf_rmse:.4f}")
        
        return (best_xgb, best_lgb, best_rf), (xgb_weight, lgb_weight, rf_weight), ensemble_rmse
    
    def feature_importance_analysis(self, model, feature_names):
        """Analyze feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\n" + "="*60)
            print("FEATURE IMPORTANCE ANALYSIS")
            print("="*60)
            print(importance_df.head(20))
            
            return importance_df
        else:
            print("Feature importance not available for this model")
            return None
    
    def model_diagnostics(self, model, X, y, model_name):
        """Perform model diagnostics"""
        print(f"\n" + "="*60)
        print(f"MODEL DIAGNOSTICS - {model_name}")
        print("="*60)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Residual analysis
        residuals = y - y_pred
        
        print(f"Residual Statistics:")
        print(f"Mean: {residuals.mean():.4f}")
        print(f"Std: {residuals.std():.4f}")
        print(f"Min: {residuals.min():.4f}")
        print(f"Max: {residuals.max():.4f}")
        
        # Prediction intervals
        print(f"\nPrediction Statistics:")
        print(f"Mean prediction: {y_pred.mean():.4f}")
        print(f"Mean actual: {y.mean():.4f}")
        print(f"Prediction range: {y_pred.min():.2f} - {y_pred.max():.2f}")
        print(f"Actual range: {y.min():.2f} - {y.max():.2f}")
    
    def run_complete_experiments(self, X, y):
        """Run complete model experimentation pipeline"""
        print("="*80)
        print("BIGMART SALES PREDICTION - MODEL EXPERIMENTS")
        print("="*80)
        
        # Prepare data
        self.prepare_data_for_experiments(X, y)
        
        # Baseline experiments
        baseline_results = self.run_baseline_experiments()
        
        # Advanced tuning and ensemble
        best_models, weights, ensemble_rmse = self.ensemble_models()
        
        # Select final model (best individual or ensemble)
        best_individual_rmse = baseline_results.iloc[0]['Val_RMSE']
        
        if ensemble_rmse < best_individual_rmse:
            print(f"\n✓ Ensemble model selected (RMSE: {ensemble_rmse:.4f})")
            final_model = best_models  # Return tuple of models for ensemble
            final_rmse = ensemble_rmse
        else:
            print(f"\n✓ Individual model selected (RMSE: {best_individual_rmse:.4f})")
            final_model = self.best_model
            final_rmse = best_individual_rmse
        
        print(f"\nFinal Model Performance: RMSE = {final_rmse:.4f}")
        
        return final_model, final_rmse, baseline_results

# Example usage
if __name__ == "__main__":
    print("BigMart Model Experiments")
    print("Note: This script requires processed features from the main solution")
    print("Run bigmart_solution.py first to prepare the data")
    
    # experiments = BigMartModelExperiments()
    # final_model, final_rmse, results = experiments.run_complete_experiments(X, y)