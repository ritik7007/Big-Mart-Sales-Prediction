# BigMart Sales Prediction - EDA and Feature Engineering Notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BigMartEDA:
    def __init__(self):
        self.train = None
        self.test = None
    
    def load_and_explore_data(self):
        """Load and perform initial exploration"""
        # Load data
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')
        
        print("="*80)
        print("BIGMART SALES PREDICTION - EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        print(f"\nDataset Shapes:")
        print(f"Train: {self.train.shape}")
        print(f"Test: {self.test.shape}")
        
        print(f"\nTrain Dataset Info:")
        print(self.train.info())
        
        print(f"\nFirst 5 rows of training data:")
        print(self.train.head())
        
        print(f"\nBasic Statistics:")
        print(self.train.describe())
        
    def missing_value_analysis(self):
        """Comprehensive missing value analysis"""
        print("\n" + "="*60)
        print("MISSING VALUE ANALYSIS")
        print("="*60)
        
        # Missing values in train
        train_missing = self.train.isnull().sum()
        train_missing_pct = (train_missing / len(self.train)) * 100
        
        print("\nMissing Values in Training Data:")
        missing_df = pd.DataFrame({
            'Column': train_missing.index,
            'Missing_Count': train_missing.values,
            'Missing_Percentage': train_missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        print(missing_df)
        
        # Missing values in test
        test_missing = self.test.isnull().sum()
        test_missing_pct = (test_missing / len(self.test)) * 100
        
        print("\nMissing Values in Test Data:")
        missing_test_df = pd.DataFrame({
            'Column': test_missing.index,
            'Missing_Count': test_missing.values,
            'Missing_Percentage': test_missing_pct.values
        })
        missing_test_df = missing_test_df[missing_test_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        print(missing_test_df)
        
    def target_variable_analysis(self):
        """Analyze the target variable distribution"""
        print("\n" + "="*60)
        print("TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        target = self.train['Item_Outlet_Sales']
        
        print(f"Target Variable Statistics:")
        print(f"Mean: {target.mean():.2f}")
        print(f"Median: {target.median():.2f}")
        print(f"Standard Deviation: {target.std():.2f}")
        print(f"Minimum: {target.min():.2f}")
        print(f"Maximum: {target.max():.2f}")
        print(f"Skewness: {target.skew():.2f}")
        print(f"Kurtosis: {target.kurtosis():.2f}")
        
        # Outlier analysis
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = target[(target < lower_bound) | (target > upper_bound)]
        print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(target)*100:.2f}%)")
        
    def categorical_variable_analysis(self):
        """Analyze categorical variables"""
        print("\n" + "="*60)
        print("CATEGORICAL VARIABLE ANALYSIS")
        print("="*60)
        
        categorical_cols = self.train.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Item_Outlet_Sales']
        
        for col in categorical_cols:
            print(f"\n{col}:")
            print(f"Unique values: {self.train[col].nunique()}")
            print(f"Value counts:")
            print(self.train[col].value_counts())
            
            # Average sales by category
            if col in self.train.columns:
                avg_sales = self.train.groupby(col)['Item_Outlet_Sales'].agg(['mean', 'median', 'count'])
                print(f"\nAverage sales by {col}:")
                print(avg_sales.sort_values('mean', ascending=False))
    
    def numerical_variable_analysis(self):
        """Analyze numerical variables"""
        print("\n" + "="*60)
        print("NUMERICAL VARIABLE ANALYSIS")
        print("="*60)
        
        numerical_cols = self.train.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'Item_Outlet_Sales']
        
        for col in numerical_cols:
            print(f"\n{col}:")
            print(f"Statistics:")
            print(self.train[col].describe())
            
            # Correlation with target
            correlation = self.train[col].corr(self.train['Item_Outlet_Sales'])
            print(f"Correlation with target: {correlation:.3f}")
            
            # Check for zeros and potential issues
            zero_count = (self.train[col] == 0).sum()
            if zero_count > 0:
                print(f"Zero values: {zero_count} ({zero_count/len(self.train)*100:.2f}%)")
    
    def data_quality_issues(self):
        """Identify data quality issues"""
        print("\n" + "="*60)
        print("DATA QUALITY ISSUES IDENTIFICATION")
        print("="*60)
        
        # 1. Item_Fat_Content inconsistencies
        print("1. Item_Fat_Content inconsistencies:")
        print(self.train['Item_Fat_Content'].value_counts())
        
        # 2. Item_Visibility = 0 (problematic)
        zero_visibility = (self.train['Item_Visibility'] == 0).sum()
        print(f"\n2. Items with 0 visibility: {zero_visibility} ({zero_visibility/len(self.train)*100:.2f}%)")
        
        # 3. Check for duplicates
        train_duplicates = self.train.duplicated().sum()
        test_duplicates = self.test.duplicated().sum()
        print(f"\n3. Duplicate rows - Train: {train_duplicates}, Test: {test_duplicates}")
        
        # 4. Item_Identifier and Outlet_Identifier combinations
        train_combinations = len(self.train[['Item_Identifier', 'Outlet_Identifier']].drop_duplicates())
        test_combinations = len(self.test[['Item_Identifier', 'Outlet_Identifier']].drop_duplicates())
        print(f"\n4. Unique Item-Outlet combinations - Train: {train_combinations}, Test: {test_combinations}")
    
    def outlet_analysis(self):
        """Detailed outlet analysis"""
        print("\n" + "="*60)
        print("OUTLET ANALYSIS")
        print("="*60)
        
        # Outlet performance
        outlet_stats = self.train.groupby('Outlet_Identifier').agg({
            'Item_Outlet_Sales': ['mean', 'median', 'sum', 'count'],
            'Item_MRP': 'mean',
            'Item_Visibility': 'mean'
        }).round(2)
        
        outlet_stats.columns = ['Avg_Sales', 'Median_Sales', 'Total_Sales', 'Item_Count', 'Avg_MRP', 'Avg_Visibility']
        outlet_stats = outlet_stats.sort_values('Avg_Sales', ascending=False)
        print("Outlet Performance Summary:")
        print(outlet_stats)
        
        # Outlet characteristics
        outlet_chars = self.train.groupby('Outlet_Identifier')[['Outlet_Size', 'Outlet_Location_Type', 
                                                               'Outlet_Type', 'Outlet_Establishment_Year']].first()
        outlet_summary = pd.merge(outlet_stats, outlet_chars, left_index=True, right_index=True)
        print("\nComplete Outlet Summary:")
        print(outlet_summary)
    
    def item_analysis(self):
        """Detailed item analysis"""
        print("\n" + "="*60)
        print("ITEM ANALYSIS")
        print("="*60)
        
        # Item type performance
        item_type_stats = self.train.groupby('Item_Type').agg({
            'Item_Outlet_Sales': ['mean', 'median', 'count'],
            'Item_MRP': 'mean',
            'Item_Weight': 'mean'
        }).round(2)
        item_type_stats.columns = ['Avg_Sales', 'Median_Sales', 'Count', 'Avg_MRP', 'Avg_Weight']
        item_type_stats = item_type_stats.sort_values('Avg_Sales', ascending=False)
        print("Item Type Performance:")
        print(item_type_stats)
        
        # Fat content analysis
        fat_content_stats = self.train.groupby('Item_Fat_Content')['Item_Outlet_Sales'].agg(['mean', 'count'])
        print("\nFat Content Analysis:")
        print(fat_content_stats)
    
    def feature_engineering_insights(self):
        """Generate insights for feature engineering"""
        print("\n" + "="*80)
        print("FEATURE ENGINEERING INSIGHTS AND RECOMMENDATIONS")
        print("="*80)
        
        print("\n1. MISSING VALUE HANDLING STRATEGY:")
        print("   - Item_Weight: Use item-specific mean (group by Item_Identifier)")
        print("   - Outlet_Size: Use mode based on Outlet_Type")
        
        print("\n2. DATA QUALITY FIXES:")
        print("   - Item_Fat_Content: Standardize inconsistent values (LF->Low Fat, reg->Regular)")
        print("   - Item_Visibility: Replace 0 values with item-specific mean")
        
        print("\n3. NEW FEATURES TO CREATE:")
        print("   - Outlet_Years: 2013 - Outlet_Establishment_Year")
        print("   - Item_MRP_Category: Categorical bins based on MRP quartiles")
        print("   - Item_Type_Combined: Group item types into broader categories")
        print("   - Visibility_Ratio: Item visibility / Average visibility of item type")
        print("   - Price_per_Weight: Item_MRP / Item_Weight")
        print("   - Outlet_Potential: Composite score based on size, type, location")
        print("   - Item_Popularity: Number of outlets selling each item")
        
        print("\n4. ENCODING STRATEGY:")
        print("   - Use Label Encoding for categorical variables")
        print("   - Consider target encoding for high-cardinality categories")
        
        print("\n5. MODEL RECOMMENDATIONS:")
        print("   - Tree-based models (Random Forest, XGBoost, LightGBM)")
        print("   - Ensemble methods for better performance")
        print("   - Cross-validation for robust evaluation")
    
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numerical columns
        numerical_cols = self.train.select_dtypes(include=[np.number]).columns
        corr_matrix = self.train[numerical_cols].corr()
        
        # Correlation with target
        target_corr = corr_matrix['Item_Outlet_Sales'].sort_values(ascending=False)
        print("Correlation with Item_Outlet_Sales:")
        print(target_corr)
        
        # High correlations between features
        print("\nHigh correlations between features (>0.5):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
    
    def run_complete_eda(self):
        """Run the complete EDA pipeline"""
        self.load_and_explore_data()
        self.missing_value_analysis()
        self.target_variable_analysis()
        self.categorical_variable_analysis()
        self.numerical_variable_analysis()
        self.data_quality_issues()
        self.outlet_analysis()
        self.item_analysis()
        self.correlation_analysis()
        self.feature_engineering_insights()
        
        print("\n" + "="*80)
        print("EDA COMPLETED - READY FOR FEATURE ENGINEERING AND MODELING")
        print("="*80)

# Run EDA
if __name__ == "__main__":
    eda = BigMartEDA()
    eda.run_complete_eda()