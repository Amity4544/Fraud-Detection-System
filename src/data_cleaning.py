"""
Data Cleaning Module
Cleans and optimizes data for processing
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning and optimization"""
    
    # Columns needed for modeling
    REQUIRED_COLUMNS = [
        'trans_date_trans_time', 'merchant', 'category', 'amt', 'gender',
        'city_pop', 'job', 'dob', 'lat', 'long', 'merch_lat', 'merch_long',
        'is_fraud'
    ]
    
    def clean_data(self, df):
        """
        Clean and optimize dataframe
        
        Args:
            df: Raw dataframe from database
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning data...")
        
        # Select required columns
        df_clean = df[self.REQUIRED_COLUMNS].copy()
        logger.info(f"Selected {len(self.REQUIRED_COLUMNS)} columns")
        
        # Convert data types
        df_clean['trans_date_trans_time'] = pd.to_datetime(df_clean['trans_date_trans_time'])
        df_clean['dob'] = pd.to_datetime(df_clean['dob'])
        
        # Convert categorical columns
        cat_cols = ['merchant', 'category', 'gender', 'job']
        for col in cat_cols:
            df_clean[col] = df_clean[col].astype('category')
        
        # Optimize numeric columns
        df_clean['is_fraud'] = df_clean['is_fraud'].astype('int8')
        df_clean['city_pop'] = pd.to_numeric(df_clean['city_pop'], downcast='integer')
        
        float_cols = ['amt', 'lat', 'long', 'merch_lat', 'merch_long']
        for col in float_cols:
            df_clean[col] = df_clean[col].astype('float32')
        
        logger.info("Data cleaning completed")
        return df_clean