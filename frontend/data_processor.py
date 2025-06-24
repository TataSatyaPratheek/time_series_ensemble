"""
Intelligent Data Processor for Real-World Time Series Forecasting
Handles complex datasets like Walmart sales data with automatic feature selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class IntelligentDataProcessor:
    """
    Intelligent processor for real-world time series data.
    Automatically detects relevant columns and handles complex datasets.
    """
    
    def __init__(self):
        self.processed_data = None
        self.metadata = {}
        self.feature_importance = {}
        
    def detect_dataset_type(self, df: pd.DataFrame) -> str:
        """Detect the type of dataset based on column patterns."""
        columns = [col.lower() for col in df.columns]
        
        if any('sales' in col or 'revenue' in col for col in columns):
            if any('store' in col for col in columns):
                return 'retail_sales'
            return 'sales'
        elif any('price' in col or 'stock' in col for col in columns):
            return 'financial'
        elif any('demand' in col or 'consumption' in col for col in columns):
            return 'demand'
        else:
            return 'generic'
    
    def detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Intelligently detect the time/date column."""
        time_candidates = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'period']):
                time_candidates.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].head(10))
                    time_candidates.append(col)
                except:
                    continue
        
        return time_candidates[0] if time_candidates else None
    
    def detect_target_column(self, df: pd.DataFrame, dataset_type: str) -> Optional[str]:
        """Intelligently detect the target column for forecasting."""
        columns = df.columns.tolist()
        
        if dataset_type == 'retail_sales':
            # Look for sales-related columns
            sales_candidates = [col for col in columns 
                              if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'amount'])]
            if sales_candidates:
                return sales_candidates[0]
        
        elif dataset_type == 'financial':
            # Look for price-related columns
            price_candidates = [col for col in columns 
                              if any(keyword in col.lower() for keyword in ['price', 'value', 'close'])]
            if price_candidates:
                return price_candidates[0]
        
        # Fallback: look for numeric columns that could be targets
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            return numeric_cols[0]
        
        return None
    
    def detect_feature_columns(self, df: pd.DataFrame, 
                             time_col: str, 
                             target_col: str,
                             dataset_type: str) -> List[str]:
        """Intelligently detect relevant feature columns."""
        all_cols = df.columns.tolist()
        exclude_cols = [time_col, target_col]
        
        # Remove identifier columns (Store, ID, etc.)
        identifier_keywords = ['id', 'store', 'shop', 'location', 'branch']
        exclude_cols.extend([col for col in all_cols 
                           if any(keyword in col.lower() for keyword in identifier_keywords)])
        
        # For retail sales, prioritize economic and environmental factors
        if dataset_type == 'retail_sales':
            priority_features = []
            feature_keywords = {
                'economic': ['cpi', 'unemployment', 'gdp', 'inflation'],
                'pricing': ['price', 'fuel', 'cost'],
                'promotional': ['markdown', 'discount', 'promotion'],
                'environmental': ['temperature', 'weather', 'holiday'],
                'temporal': ['month', 'quarter', 'season', 'week']
            }
            
            for category, keywords in feature_keywords.items():
                for col in all_cols:
                    if (col not in exclude_cols and 
                        any(keyword in col.lower() for keyword in keywords)):
                        priority_features.append(col)
            
            # Add other numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col not in exclude_cols and col not in priority_features:
                    priority_features.append(col)
            
            return priority_features
        
        # Generic approach: return all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in exclude_cols]
    
    def process_walmart_data(self, 
                           features_df: pd.DataFrame,
                           stores_df: pd.DataFrame,
                           sales_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Specific processor for Walmart-style retail data.
        """
        st.info("🏪 Detected Walmart-style retail dataset. Processing...")
        
        # If sales data is separate, merge it
        if sales_df is not None:
            # Merge sales with features
            merged_df = pd.merge(sales_df, features_df, on=['Store', 'Date'], how='inner')
        else:
            # Assume features_df contains sales data
            merged_df = features_df.copy()
        
        # Merge with store information
        if 'Store' in merged_df.columns and 'Store' in stores_df.columns:
            merged_df = pd.merge(merged_df, stores_df, on='Store', how='left')
        
        # Process the merged data
        processed_df = self._process_retail_data(merged_df)
        
        return processed_df
    
    def _process_retail_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process retail-specific data transformations."""
        processed_df = df.copy()
        
        # Convert Date column
        date_col = self.detect_time_column(processed_df)
        if date_col:
            processed_df[date_col] = pd.to_datetime(processed_df[date_col])
            processed_df = processed_df.sort_values(date_col)
        
        # Handle missing values intelligently
        processed_df = self._handle_missing_values(processed_df)
        
        # Create time-based features
        if date_col:
            processed_df = self._create_temporal_features(processed_df, date_col)
        
        # Handle categorical variables
        processed_df = self._encode_categorical_features(processed_df)
        
        return processed_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing value handling."""
        processed_df = df.copy()
        
        for col in processed_df.columns:
            if processed_df[col].isnull().sum() > 0:
                missing_pct = processed_df[col].isnull().sum() / len(processed_df) * 100
                
                if missing_pct > 50:
                    # Drop columns with too many missing values
                    st.warning(f"Dropping column '{col}' due to {missing_pct:.1f}% missing values")
                    processed_df = processed_df.drop(columns=[col])
                    continue
                
                if processed_df[col].dtype in ['int64', 'float64']:
                    # Forward fill for numeric time series
                    processed_df[col] = processed_df[col].fillna(method='ffill')
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                else:
                    # Most frequent for categorical
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode().iloc[0] if not processed_df[col].mode().empty else 'Unknown')
        
        return processed_df
    
    def _create_temporal_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create time-based features."""
        processed_df = df.copy()
        
        processed_df['Year'] = processed_df[date_col].dt.year
        processed_df['Month'] = processed_df[date_col].dt.month
        processed_df['Quarter'] = processed_df[date_col].dt.quarter
        processed_df['Week'] = processed_df[date_col].dt.isocalendar().week
        processed_df['DayOfYear'] = processed_df[date_col].dt.dayofyear
        processed_df['IsYearEnd'] = processed_df[date_col].dt.is_year_end.astype(int)
        processed_df['IsQuarterEnd'] = processed_df[date_col].dt.is_quarter_end.astype(int)
        
        # Holiday effects (if IsHoliday column exists)
        if 'IsHoliday' in processed_df.columns:
            processed_df['IsHoliday'] = processed_df['IsHoliday'].astype(int)
        
        return processed_df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        processed_df = df.copy()
        
        categorical_cols = processed_df.select_dtypes(include=['object', 'bool']).columns
        
        for col in categorical_cols:
            if col.lower() in ['date', 'timestamp']:
                continue
                
            unique_values = processed_df[col].nunique()
            
            if unique_values <= 10:  # One-hot encode for low cardinality
                dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
                processed_df = pd.concat([processed_df, dummies], axis=1)
                processed_df = processed_df.drop(columns=[col])
            else:  # Label encode for high cardinality
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
        
        return processed_df
    
    def aggregate_to_time_series(self, 
                                df: pd.DataFrame,
                                time_col: str,
                                target_col: str,
                                feature_cols: List[str],
                                agg_level: str = 'weekly') -> pd.DataFrame:
        """
        Aggregate data to create a proper time series.
        """
        # Set up grouping
        df[time_col] = pd.to_datetime(df[time_col])
        
        if agg_level == 'weekly':
            df['Period'] = df[time_col].dt.to_period('W')
        elif agg_level == 'monthly':
            df['Period'] = df[time_col].dt.to_period('M')
        elif agg_level == 'daily':
            df['Period'] = df[time_col].dt.to_period('D')
        else:
            df['Period'] = df[time_col].dt.to_period('W')  # Default to weekly
        
        # Aggregation rules
        agg_dict = {}
        
        # Target column - sum for sales
        if 'sales' in target_col.lower() or 'revenue' in target_col.lower():
            agg_dict[target_col] = 'sum'
        else:
            agg_dict[target_col] = 'mean'
        
        # Feature columns
        for col in feature_cols:
            if col in df.columns:
                if any(keyword in col.lower() for keyword in ['price', 'cpi', 'unemployment', 'temperature']):
                    agg_dict[col] = 'mean'  # Average for rates and prices
                elif any(keyword in col.lower() for keyword in ['markdown', 'discount']):
                    agg_dict[col] = 'sum'   # Sum for promotional spending
                else:
                    agg_dict[col] = 'mean'  # Default to mean
        
        # Perform aggregation
        if 'Store' in df.columns:
            # Aggregate across all stores for each time period
            aggregated = df.groupby('Period').agg(agg_dict).reset_index()
        else:
            aggregated = df.groupby('Period').agg(agg_dict).reset_index()
        
        # Convert Period back to datetime
        aggregated['Date'] = aggregated['Period'].dt.start_time
        aggregated = aggregated.drop(columns=['Period'])
        aggregated = aggregated.sort_values('Date').reset_index(drop=True)
        
        return aggregated
    
    def create_forecasting_dataset(self, 
                                 uploaded_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Main function to create a forecasting-ready dataset from uploaded files.
        """
        try:
            # Handle different file combinations
            if len(uploaded_files) == 1:
                # Single file - assume it's complete
                df = list(uploaded_files.values())[0]
                dataset_type = self.detect_dataset_type(df)
                
            elif 'features.csv' in uploaded_files:
                # Multi-file Walmart-style dataset
                features_df = uploaded_files['features.csv']
                stores_df = uploaded_files.get('stores.csv')
                sales_df = uploaded_files.get('sales.csv')  # If separate sales file
                
                if stores_df is not None:
                    df = self.process_walmart_data(features_df, stores_df, sales_df)
                else:
                    df = features_df
                dataset_type = 'retail_sales'
            
            else:
                # Merge all files if they have common columns
                df = list(uploaded_files.values())[0]
                for file_df in list(uploaded_files.values())[1:]:
                    # Try to find common columns for merging
                    common_cols = set(df.columns) & set(file_df.columns)
                    if common_cols:
                        df = pd.merge(df, file_df, on=list(common_cols), how='outer')
                dataset_type = self.detect_dataset_type(df)
            
            # Detect key columns
            time_col = self.detect_time_column(df)
            target_col = self.detect_target_column(df, dataset_type)
            feature_cols = self.detect_feature_columns(df, time_col, target_col, dataset_type)
            
            # Store metadata
            self.metadata = {
                'dataset_type': dataset_type,
                'time_column': time_col,
                'target_column': target_col,
                'feature_columns': feature_cols,
                'original_shape': df.shape,
                'date_range': (df[time_col].min(), df[time_col].max()) if time_col else None
            }
            
            # Select only relevant columns
            selected_cols = [col for col in [time_col, target_col] + feature_cols if col and col in df.columns]
            df_selected = df[selected_cols].copy()
            
            # Create aggregated time series
            if dataset_type == 'retail_sales' and 'Store' in df.columns:
                # For retail data, aggregate across stores
                final_df = self.aggregate_to_time_series(
                    df, time_col, target_col, feature_cols, 'weekly'
                )
            else:
                # For other data types, use as-is but ensure proper time series format
                final_df = df_selected.copy()
                if time_col:
                    final_df[time_col] = pd.to_datetime(final_df[time_col])
                    final_df = final_df.sort_values(time_col).reset_index(drop=True)
            
            # Final cleaning
            final_df = self._handle_missing_values(final_df)
            
            self.processed_data = final_df
            self.metadata['processed_shape'] = final_df.shape
            
            return final_df
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return pd.DataFrame()
    
    def get_forecasting_series(self, target_col: str = None) -> pd.Series:
        """Extract the main time series for forecasting."""
        if self.processed_data is None:
            return pd.Series()
        
        if target_col is None:
            target_col = self.metadata.get('target_column')
        
        if target_col and target_col in self.processed_data.columns:
            time_col = self.metadata.get('time_column')
            if time_col and time_col in self.processed_data.columns:
                return pd.Series(
                    self.processed_data[target_col].values,
                    index=pd.to_datetime(self.processed_data[time_col]),
                    name=target_col
                )
            else:
                return pd.Series(self.processed_data[target_col].values, name=target_col)
        
        return pd.Series()
    
    def display_data_summary(self):
        """Display summary of processed data."""
        if self.processed_data is None or self.metadata == {}:
            st.warning("No data processed yet.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset Type", self.metadata['dataset_type'].replace('_', ' ').title())
            st.metric("Records", f"{self.metadata['processed_shape'][0]:,}")
        
        with col2:
            st.metric("Features", f"{len(self.metadata['feature_columns'])}")
            st.metric("Target", self.metadata['target_column'])
        
        with col3:
            if self.metadata['date_range']:
                date_range = self.metadata['date_range']
                st.metric("Date Range", f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
        
        # Show column selection
        with st.expander("📊 Column Selection Details"):
            st.write("**Selected Columns:**")
            
            col_info = pd.DataFrame({
                'Column': [self.metadata['time_column']] + [self.metadata['target_column']] + self.metadata['feature_columns'],
                'Type': ['Time'] + ['Target'] + ['Feature'] * len(self.metadata['feature_columns'])
            })
            st.dataframe(col_info, use_container_width=True)
    
    def create_data_visualization(self):
        """Create visualization of the processed data."""
        if self.processed_data is None:
            return
        
        target_col = self.metadata['target_column']
        time_col = self.metadata['time_column']
        
        if not target_col or target_col not in self.processed_data.columns:
            return
        
        # Create subplots
        feature_cols = self.metadata['feature_columns'][:4]  # Show top 4 features
        n_plots = len(feature_cols) + 1
        
        fig = make_subplots(
            rows=n_plots, cols=1,
            subplot_titles=[f'{target_col} (Target)'] + [f'{col} (Feature)' for col in feature_cols],
            vertical_spacing=0.08
        )
        
        # Target series
        if time_col:
            x_data = self.processed_data[time_col]
        else:
            x_data = self.processed_data.index
        
        fig.add_trace(
            go.Scatter(x=x_data, y=self.processed_data[target_col], name=target_col, line=dict(color='blue')),
            row=1, col=1
        )
        
        # Feature series
        colors = ['red', 'green', 'orange', 'purple']
        for i, col in enumerate(feature_cols):
            if col in self.processed_data.columns:
                fig.add_trace(
                    go.Scatter(x=x_data, y=self.processed_data[col], name=col, line=dict(color=colors[i % len(colors)])),
                    row=i+2, col=1
                )
        
        fig.update_layout(height=200*n_plots, showlegend=False, title="Processed Data Overview")
        
        st.plotly_chart(fig, use_container_width=True)
