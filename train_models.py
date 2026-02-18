"""
Model Training Script
Trains multiple models and saves the best one
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
os.makedirs('results/plots', exist_ok=True)


def load_preprocessed_data():
    """Load preprocessed data from files"""
    logger.info("Loading preprocessed data...")
    
    if not os.path.exists('data/X_train.pkl'):
        raise FileNotFoundError("Preprocessed data not found. Run main.py first!")
    
    X_train = joblib.load('data/X_train.pkl')
    y_train = joblib.load('data/y_train.pkl')
    X_test = joblib.load('data/X_test.pkl')
    y_test = joblib.load('data/y_test.pkl')
    
    logger.info(f"✅ Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def apply_smote_tomek(X_train, y_train):
    """Apply SMOTE-Tomek to handle class imbalance"""
    logger.info("\nApplying SMOTE-Tomek...")
    logger.info(f"Before: Total={len(y_train):,}, Fraud={sum(y_train):,}")
    
    smt = SMOTETomek(
        sampling_strategy=0.2,
        smote=SMOTE(sampling_strategy=0.2, random_state=42, n_jobs=-1),
        tomek=TomekLinks(n_jobs=-1),
        random_state=42
    )
    
    X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
    
    logger.info(f"After:  Total={len(y_resampled):,}, Fraud={sum(y_resampled):,}")
    logger.info("✅ Resampling completed")
    
    return X_resampled, y_resampled


def train_all_models(X_train, y_train, X_test, y_test):
    """Train all models and evaluate"""
    logger.info("\n" + "="*60)
    logger.info("Training 6 models...")
    logger.info("="*60)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, n_jobs=-1, 
                                eval_metric='logloss', random_state=42),
        "CatBoost": CatBoostClassifier(iterations=500, depth=8, learning_rate=0.1, 
                                      l2_leaf_reg=1, verbose=0, thread_count=-1, random_state=42),
    }
    
    results = []
    trained_models = {}
    confusion_matrices = {}
    
    for name, model in models.items():
        logger.info(f"\n🔄 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        auprc = average_precision_score(y_test, y_probs)
        
        # Store results
        results.append({
            'Model': name,
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1-Score': report['1']['f1-score'],
            'AUPRC': auprc
        })
        
        trained_models[name] = model
        confusion_matrices[name] = cm
        
        logger.info(f"   ✅ AUPRC: {auprc:.4f}, Precision: {report['1']['precision']:.2%}, Recall: {report['1']['recall']:.2%}")
    
    return results, trained_models, confusion_matrices


def save_results(results, confusion_matrices):
    """Save results and plots"""
    logger.info("\n" + "="*60)
    logger.info("Saving results...")
    logger.info("="*60)
    
    # Save comparison table
    df = pd.DataFrame(results).sort_values('AUPRC', ascending=False)
    df.to_csv('results/model_comparison.csv', index=False)
    logger.info("✅ Saved: results/model_comparison.csv")
    
    # Print table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('results/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
    logger.info("✅ Saved: results/plots/confusion_matrices.png")
    plt.close()


def save_best_model(results, trained_models):
    """Save the best model"""
    logger.info("\nSaving best model...")
    
    # Find best model by AUPRC
    best_idx = max(range(len(results)), key=lambda i: results[i]['AUPRC'])
    best_name = results[best_idx]['Model']
    best_model = trained_models[best_name]
    
    # Save
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Save metadata
    metadata = {
        'model_name': best_name,
        'metrics': results[best_idx]
    }
    joblib.dump(metadata, 'models/model_metadata.pkl')
    
    logger.info(f"✅ Best Model: {best_name}")
    logger.info(f"   AUPRC: {results[best_idx]['AUPRC']:.4f}")
    logger.info(f"   Precision: {results[best_idx]['Precision']:.2%}")
    logger.info(f"   Recall: {results[best_idx]['Recall']:.2%}")
    logger.info("✅ Saved: models/best_model.pkl")
    
    return best_name


def main():
    """Run complete training pipeline"""
    
    print("\n" + "="*60)
    print("  FRAUD DETECTION - MODEL TRAINING")
    print("="*60 + "\n")
    
    try:
        # Load data
        X_train, y_train, X_test, y_test = load_preprocessed_data()
        
        # Apply SMOTE-Tomek
        X_train_res, y_train_res = apply_smote_tomek(X_train, y_train)
        
        # Train models
        results, trained_models, confusion_matrices = train_all_models(
            X_train_res, y_train_res, X_test, y_test
        )
        
        # Save results
        save_results(results, confusion_matrices)
        
        # Save best model
        best_name = save_best_model(results, trained_models)
        
        # Success message
        print("\n" + "="*60)
        print("  ✅ TRAINING COMPLETED!")
        print("="*60)
        print("\n📁 Saved Files:")
        print("  ✅ models/best_model.pkl")
        print("  ✅ results/model_comparison.csv")
        print("  ✅ results/plots/confusion_matrices.png")
        print(f"\n🏆 Best Model: {best_name}")
        print("\n🚀 Next Step:")
        print("  Run: python app.py")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()