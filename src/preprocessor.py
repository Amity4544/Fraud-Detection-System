"""
Preprocessor Module
Handles encoding and scaling of features
"""

import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Preprocessor:
    """Handles feature preprocessing (encoding and scaling)"""
    
    CATEGORICAL_COLUMNS = ['merchant', 'category', 'gender', 'job']
    
    def __init__(self, models_dir='models'):
        """Initialize preprocessor"""
        self.models_dir = models_dir
        self.encoder = None
        self.scaler = None
        os.makedirs(self.models_dir, exist_ok=True)
    
    def prepare_train_data(self, df):
        """
        Prepare training data (fit and transform)
        
        Args:
            df: Dataframe with engineered features
            
        Returns:
            tuple: (X_train, y_train)
        """
        logger.info("Preparing training data...")
        
        # Separate features and target
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']
        
        # Fit and transform encoder
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[self.CATEGORICAL_COLUMNS] = self.encoder.fit_transform(X[self.CATEGORICAL_COLUMNS])
        
        # Fit and transform scaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Save transformers
        joblib.dump(self.encoder, os.path.join(self.models_dir, 'encoder.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        logger.info("Transformers saved")
        
        logger.info(f"Training data prepared: {X_scaled.shape}")
        return X_scaled, y
    
    def prepare_test_data(self, df):
        """
        Prepare test data (transform only)
        
        Args:
            df: Dataframe with engineered features
            
        Returns:
            tuple: (X_test, y_test)
        """
        logger.info("Preparing test data...")
        
        if self.encoder is None or self.scaler is None:
            raise ValueError("Transformers not fitted. Call prepare_train_data first.")
        
        # Separate features and target
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']
        
        # Transform using fitted encoder and scaler
        X[self.CATEGORICAL_COLUMNS] = self.encoder.transform(X[self.CATEGORICAL_COLUMNS])
        X_scaled = self.scaler.transform(X)
        
        logger.info(f"Test data prepared: {X_scaled.shape}")
        return X_scaled, y