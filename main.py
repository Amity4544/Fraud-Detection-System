"""
Main Data Pipeline
Runs: Extract → Clean → Engineer → Preprocess
Saves: Preprocessed data as PKL files
"""

import os
import joblib
import logging
from src.data_loader import DataLoader
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)


def main():
    """Run complete data pipeline"""
    
    print("\n" + "="*60)
    print("  FRAUD DETECTION - DATA PIPELINE")
    print("="*60)
    
    # Check if preprocessed data already exists
    if os.path.exists('data/X_train.pkl') and os.path.exists('data/X_test.pkl'):
        logger.info("\n✅ Preprocessed data already exists!")
        logger.info("Files found:")
        logger.info("  - data/X_train.pkl")
        logger.info("  - data/y_train.pkl")
        logger.info("  - data/X_test.pkl")
        logger.info("  - data/y_test.pkl")
        logger.info("\nSkipping pipeline. To re-run, delete these files.")
        print("\n" + "="*60)
        return
    
    try:
        # Step 1: Load data from MySQL
        logger.info("\n📥 STEP 1: Loading data from MySQL...")
        loader = DataLoader()
        train_df, test_df = loader.load_train_test_data()
        
        # Step 2: Clean data
        logger.info("\n🧹 STEP 2: Cleaning data...")
        cleaner = DataCleaner()
        train_clean = cleaner.clean_data(train_df)
        test_clean = cleaner.clean_data(test_df)
        
        # Step 3: Engineer features
        logger.info("\n🔧 STEP 3: Engineering features...")
        engineer = FeatureEngineer()
        train_featured = engineer.engineer_features(train_clean)
        test_featured = engineer.engineer_features(test_clean)
        
        # Step 4: Preprocess (encode & scale)
        logger.info("\n⚙️  STEP 4: Preprocessing (encoding & scaling)...")
        preprocessor = Preprocessor()
        X_train, y_train = preprocessor.prepare_train_data(train_featured)
        X_test, y_test = preprocessor.prepare_test_data(test_featured)
        
        # Step 5: Save preprocessed data
        logger.info("\n💾 STEP 5: Saving preprocessed data...")
        joblib.dump(X_train, 'data/X_train.pkl')
        joblib.dump(y_train, 'data/y_train.pkl')
        joblib.dump(X_test, 'data/X_test.pkl')
        joblib.dump(y_test, 'data/y_test.pkl')
        
        # Success message
        print("\n" + "="*60)
        print("  ✅ DATA PIPELINE COMPLETED!")
        print("="*60)
        print("\n📁 Saved Files:")
        print("  ✅ data/X_train.pkl")
        print("  ✅ data/y_train.pkl")
        print("  ✅ data/X_test.pkl")
        print("  ✅ data/y_test.pkl")
        print("  ✅ models/encoder.pkl")
        print("  ✅ models/scaler.pkl")
        print("\n📊 Data Info:")
        print(f"  Training samples: {len(y_train):,}")
        print(f"  Test samples: {len(y_test):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Fraud rate (train): {y_train.mean()*100:.2f}%")
        print("\n🚀 Next Step:")
        print("  Run: python train_models.py")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()