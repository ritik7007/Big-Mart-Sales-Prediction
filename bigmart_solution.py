import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

class BigMartSalesPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        
    def load_data(self):
        """Load train and test datasets"""
        try:
            self.train = pd.read_csv('train_v9rqX0R.csv')
            self.test = pd.read_csv('test_AbJTz2l.csv')
            print("Data loaded successfully!")
            print(f"Train shape: {self.train.shape}")
            print(f"Test shape: {self.test.shape}")
            return True
        except FileNotFoundError:
            print("Error: train.csv or test.csv not found!")
            return False
    
    def exploratory_data_analysis(self):
        """Comprehensive EDA"""
        print("="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic info
        print("\nTrain Dataset Info:")
        print(self.train.info())
        print("\nTrain Dataset Description:")
        print(self.train.describe())
        
        # Missing values
        print("\nMissing Values in Train:")
        missing_train = self.train.isnull().sum()
        print(missing_train[missing_train > 0])
        
        print("\nMissing Values in Test:")
        missing_test = self.test.isnull().sum()
        print(missing_test[missing_test > 0])
        
        # Target variable distribution
        print(f"\nTarget Variable Statistics:")
        print(f"Mean Sales: {self.train['Item_Outlet_Sales'].mean():.2f}")
        print(f"Median Sales: {self.train['Item_Outlet_Sales'].median():.2f}")
        print(f"Standard Deviation: {self.train['Item_Outlet_Sales'].std():.2f}")
        
        # Categorical variables
        categorical_cols = self.train.select_dtypes(include=['object']).columns
        print(f"\nCategorical Columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            print(f"\n{col} - Unique Values: {self.train[col].nunique()}")
            print(self.train[col].value_counts().head())
    
    def advanced_feature_engineering(self):
        """Advanced feature engineering"""
        print("\n" + "="*60)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*60)
        
        # Combine train and test for consistent preprocessing
        self.train['source'] = 'train'
        self.test['source'] = 'test'
        combined = pd.concat([self.train, self.test], ignore_index=True)
        
        # 1. Handle Item_Fat_Content inconsistencies
        combined['Item_Fat_Content'] = combined['Item_Fat_Content'].replace({
            'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'
        })
        
        # 2. Create Item_Type_Combined (broader categories)
        combined['Item_Type_Combined'] = combined['Item_Type'].apply(lambda x: 
            'Food' if x in ['Dairy', 'Meat', 'Fruits and Vegetables', 'Snack Foods', 
                           'Breakfast', 'Frozen Foods', 'Canned', 'Baking Goods'] 
            else 'Non-Food')
        
        # 3. Handle missing Item_Weight
        item_weight_mean = combined.groupby('Item_Identifier')['Item_Weight'].mean()
        combined['Item_Weight'] = combined.apply(
            lambda x: item_weight_mean[x['Item_Identifier']] 
            if pd.isna(x['Item_Weight']) else x['Item_Weight'], axis=1
        )
        
        # 4. Handle missing Outlet_Size
        # Use mode based on Outlet_Type
        outlet_size_mode = combined.groupby('Outlet_Type')['Outlet_Size'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Medium'
        )
        combined['Outlet_Size'] = combined.apply(
            lambda x: outlet_size_mode[x['Outlet_Type']] 
            if pd.isna(x['Outlet_Size']) else x['Outlet_Size'], axis=1
        )
        
        # 5. Fix Item_Visibility (0 values are problematic)
        item_visibility_mean = combined.groupby('Item_Identifier')['Item_Visibility'].mean()
        combined['Item_Visibility'] = combined.apply(
            lambda x: item_visibility_mean[x['Item_Identifier']] 
            if x['Item_Visibility'] == 0 else x['Item_Visibility'], axis=1
        )
        
        # 6. Create new features
        # Years of operation
        combined['Outlet_Years'] = 2013 - combined['Outlet_Establishment_Year']
        
        # Item_MRP categories
        combined['Item_MRP_Category'] = pd.cut(combined['Item_MRP'], 
                                             bins=[0, 69, 136, 203, 270], 
                                             labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Visibility ratio (compared to average visibility of item type)
        visibility_by_type = combined.groupby('Item_Type')['Item_Visibility'].mean()
        combined['Visibility_Ratio'] = combined.apply(
            lambda x: x['Item_Visibility'] / visibility_by_type[x['Item_Type']], axis=1
        )
        
        # Price per unit weight
        combined['Price_per_Weight'] = combined['Item_MRP'] / combined['Item_Weight']
        
        # Outlet sales potential (based on size and location)
        outlet_mapping = {'Grocery Store': 1, 'Supermarket Type1': 2, 
                         'Supermarket Type2': 3, 'Supermarket Type3': 4}
        combined['Outlet_Type_Numeric'] = combined['Outlet_Type'].map(outlet_mapping)
        
        size_mapping = {'Small': 1, 'Medium': 2, 'High': 3}
        combined['Outlet_Size_Numeric'] = combined['Outlet_Size'].map(size_mapping)
        
        location_mapping = {'Tier 3': 1, 'Tier 2': 2, 'Tier 1': 3}
        combined['Outlet_Location_Numeric'] = combined['Outlet_Location_Type'].map(location_mapping)
        
        # Outlet potential score
        combined['Outlet_Potential'] = (combined['Outlet_Type_Numeric'] * 0.4 + 
                                       combined['Outlet_Size_Numeric'] * 0.3 + 
                                       combined['Outlet_Location_Numeric'] * 0.3)
        
        # Item popularity (based on number of outlets selling it)
        item_popularity = combined.groupby('Item_Identifier')['Outlet_Identifier'].count()
        combined['Item_Popularity'] = combined['Item_Identifier'].map(item_popularity)
        
        # Split back to train and test
        self.train_processed = combined[combined['source'] == 'train'].drop('source', axis=1)
        self.test_processed = combined[combined['source'] == 'test'].drop(['source', 'Item_Outlet_Sales'], axis=1)
        
        print("Feature engineering completed!")
        print(f"New train shape: {self.train_processed.shape}")
        print(f"New test shape: {self.test_processed.shape}")
        
        return self.train_processed, self.test_processed
    
    def prepare_features(self):
        """Prepare features for modeling"""
        # Define features to drop
        features_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']
        
        # Categorical features to encode
        categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
                               'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_Combined',
                               'Item_MRP_Category']
        
        # Prepare training data
        X = self.train_processed.drop(features_to_drop, axis=1)
        y = self.train_processed['Item_Outlet_Sales']
        
        # Prepare test data
        X_test = self.test_processed.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)
        
        # Label encoding for categorical variables
        for col in categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                self.label_encoders[col] = le
        
        self.feature_names = X.columns.tolist()
        
        return X, y, X_test
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("\n" + "="*60)
        print("MODEL TRAINING AND EVALUATION")
        print("="*60)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Models to try
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42)
        }
        
        results = {}
        best_rmse = float('inf')
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                results[name] = {'RMSE': rmse, 'R2': r2, 'Model': model}
                
                print(f"{name}: RMSE = {rmse:.2f}, R2 = {r2:.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        print(f"\nBest Model: {self.best_model_name} with RMSE: {best_rmse:.2f}")
        
        return results
    
    def hyperparameter_tuning(self, X, y):
        """Advanced hyperparameter tuning for the best models"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # XGBoost hyperparameter tuning
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, 
                               scoring='neg_mean_squared_error', n_jobs=-1)
        xgb_grid.fit(X, y)
        
        # Random Forest hyperparameter tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_model = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf_model, rf_params, cv=5, 
                              scoring='neg_mean_squared_error', n_jobs=-1)
        rf_grid.fit(X, y)
        
        # Compare tuned models
        xgb_rmse = np.sqrt(-xgb_grid.best_score_)
        rf_rmse = np.sqrt(-rf_grid.best_score_)
        
        print(f"XGBoost Best RMSE: {xgb_rmse:.2f}")
        print(f"Random Forest Best RMSE: {rf_rmse:.2f}")
        
        if xgb_rmse < rf_rmse:
            self.best_model = xgb_grid.best_estimator_
            self.best_model_name = "Tuned XGBoost"
            best_rmse = xgb_rmse
        else:
            self.best_model = rf_grid.best_estimator_
            self.best_model_name = "Tuned Random Forest"
            best_rmse = rf_rmse
        
        print(f"Final Best Model: {self.best_model_name} with RMSE: {best_rmse:.2f}")
        
        return self.best_model
    
    def make_predictions(self, X_test):
        """Make predictions on test set"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        predictions = self.best_model.predict(X_test)
        
        # Ensure no negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def create_submission(self, predictions):
        """Create submission file"""
        submission = pd.DataFrame({
            'Item_Identifier': self.test['Item_Identifier'],
            'Outlet_Identifier': self.test['Outlet_Identifier'],
            'Item_Outlet_Sales': predictions
        })
        
        submission.to_csv('submission.csv', index=False)
        print("Submission file created: submission.csv")
        
        return submission
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 15 Most Important Features:")
            print(importance_df.head(15))
            
            return importance_df
        else:
            print("Feature importance not available for this model type.")
            return None
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("BigMart Sales Prediction - Complete Pipeline")
        print("="*60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: EDA
        self.exploratory_data_analysis()
        
        # Step 3: Feature Engineering
        self.advanced_feature_engineering()
        
        # Step 4: Prepare features
        X, y, X_test = self.prepare_features()
        
        # Step 5: Train models
        results = self.train_models(X, y)
        
        # Step 6: Hyperparameter tuning
        self.hyperparameter_tuning(X, y)
        
        # Step 7: Feature importance
        self.feature_importance_analysis()
        
        # Step 8: Make predictions
        predictions = self.make_predictions(X_test)
        
        # Step 9: Create submission
        submission = self.create_submission(predictions)
        
        print("\nPipeline completed successfully!")
        print(f"Final predictions summary:")
        print(f"Mean prediction: {predictions.mean():.2f}")
        print(f"Min prediction: {predictions.min():.2f}")
        print(f"Max prediction: {predictions.max():.2f}")
        
        return True

# Main execution
if __name__ == "__main__":
    predictor = BigMartSalesPredictor()
    success = predictor.run_complete_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS! All files generated:")
        print("1. submission.csv - Final predictions for submission")
        print("2. This script contains all the modeling code")
        print("="*60)
    else:
        print("Pipeline failed. Please check if train.csv and test.csv files are present.")