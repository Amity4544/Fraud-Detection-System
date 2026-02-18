"""
Data Loader Module
Extracts data from MySQL database
"""

import pandas as pd
import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data extraction from MySQL database"""
    
    def __init__(self):
        """Initialize database connection parameters"""
        self.db_config = {
            'drivername': 'mysql+pymysql',
            'username': os.getenv('DB_USERNAME', 'root'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'database': os.getenv('DB_NAME', 'fraud_detection')
        }
        
        if not self.db_config['password']:
            raise ValueError("Database password not found. Set DB_PASSWORD in .env file")
    
    def load_train_test_data(self):
        """
        Load training and testing data from database
        
        Returns:
            tuple: (train_df, test_df)
        """
        logger.info("Connecting to MySQL database...")
        
        try:
            connection_url = URL.create(**self.db_config)
            engine = create_engine(connection_url)
            
            # Load training data
            query_train = "SELECT * FROM train_table"
            train_df = pd.read_sql(query_train, con=engine)
            logger.info(f"Loaded training data: {train_df.shape}")
            
            # Load testing data
            query_test = "SELECT * FROM test_table"
            test_df = pd.read_sql(query_test, con=engine)
            logger.info(f"Loaded testing data: {test_df.shape}")
            
            engine.dispose()
            logger.info("Database connection closed")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise