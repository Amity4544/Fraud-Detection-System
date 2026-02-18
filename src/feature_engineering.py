"""
Feature Engineering Module
Creates new features from existing data
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for fraud detection"""
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points using Haversine formula
        
        Args:
            lat1, lon1: Customer coordinates
            lat2, lon2: Merchant coordinates
            
        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def engineer_features(self, df):
        """
        Create new features from existing columns
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Dataframe with new features
        """
        logger.info("Engineering features...")
        
        df_featured = df.copy()
        
        # Time-based features
        df_featured['hour'] = df_featured['trans_date_trans_time'].dt.hour
        df_featured['day_of_week'] = df_featured['trans_date_trans_time'].dt.dayofweek
        df_featured['is_night'] = df_featured['hour'].apply(lambda x: 1 if x <= 4 or x >= 22 else 0)
        
        # Age feature
        df_featured['age'] = (df_featured['trans_date_trans_time'] - df_featured['dob']).dt.days // 365
        
        # Distance feature
        df_featured['dist_to_merchant'] = self.haversine_distance(
            df_featured['lat'], df_featured['long'],
            df_featured['merch_lat'], df_featured['merch_long']
        )
        
        # Drop original columns that were used to create features
        cols_to_drop = ['trans_date_trans_time', 'dob', 'lat', 'long', 'merch_lat', 'merch_long']
        df_featured = df_featured.drop(columns=cols_to_drop)
        
        logger.info(f"Feature engineering completed. Final shape: {df_featured.shape}")
        return df_featured