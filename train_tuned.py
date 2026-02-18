"""
Hyperparameter Tuning Script
Uses RandomizedSearchCV to optimize top 2 models (CatBoost & XGBoost)
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, make_scorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs('results/plots', exist_ok=True)


def load_preprocessed_data():
    """Load preprocessed data"""
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
    """Apply SMOTE-Tomek"""
    logger.info("\nApplying SMOTE-Tomek...")
    logger.info(f"Before: Total={len(y_train):,}, Fraud={sum(y_train):,}")
    
    smt = SMOTETomek(
        sampling_strategy=0.2,
        smote=SMOTE(sampling_strategy=0.2, random_state=42),
        tomek=TomekLinks(),
        random_state=42
    )
    
    X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
    
    logger.info(f"After:  Total={len(y_resampled):,}, Fraud={sum(y_resampled):,}")
    logger.info("✅ Resampling completed")
    
    return X_resampled, y_resampled


def train_baseline_models(X_train, y_train, X_test, y_test):
    """Train baseline models without tuning"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Training baseline models...")
    logger.info("="*60)
    
    baseline_models = {
        "XGBoost (Baseline)": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            n_jobs=-1,
            eval_metric='logloss',
            random_state=42
        ),
        "CatBoost (Baseline)": CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.1,
            l2_leaf_reg=1,
            verbose=0,
            thread_count=-1,
            random_state=42
        )
    }
    
    baseline_results = {}
    
    for name, model in baseline_models.items():
        logger.info(f"\n🔄 Training {name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auprc = average_precision_score(y_test, y_probs)
        
        baseline_results[name] = {
            'model': model,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'auprc': auprc
        }
        
        logger.info(f"   ✅ Precision: {report['1']['precision']:.2%}, Recall: {report['1']['recall']:.2%}, AUPRC: {auprc:.4f}")
    
    return baseline_results


def tune_xgboost(X_train, y_train, X_test, y_test):
    """Tune XGBoost with RandomizedSearchCV"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Tuning XGBoost with RandomizedSearchCV...")
    logger.info("="*60)
    
    # Parameter grid
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    logger.info("Parameter grid:")
    logger.info(f"  - n_estimators: {param_dist['n_estimators']}")
    logger.info(f"  - max_depth: {param_dist['max_depth']}")
    logger.info(f"  - learning_rate: {param_dist['learning_rate']}")
    logger.info(f"  Total combinations: {np.prod([len(v) for v in param_dist.values()]):,}")
    logger.info(f"  Testing: 30 random combinations with 3-fold CV")
    
    # RandomizedSearchCV
    xgb_base = XGBClassifier(
        n_jobs=-1,
        eval_metric='logloss',
        random_state=42
    )
    
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring=make_scorer(average_precision_score, needs_proba=True),
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    logger.info("\n⏳ Starting RandomizedSearchCV (this will take 20-30 minutes)...")
    random_search.fit(X_train, y_train)
    
    # Best model
    best_xgb = random_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    y_probs = best_xgb.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auprc = average_precision_score(y_test, y_probs)
    
    logger.info("\n✅ XGBoost Tuning Complete!")
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best CV score: {random_search.best_score_:.4f}")
    logger.info(f"Test - Precision: {report['1']['precision']:.2%}, Recall: {report['1']['recall']:.2%}, AUPRC: {auprc:.4f}")
    
    return {
        'model': best_xgb,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'auprc': auprc,
        'best_params': random_search.best_params_
    }


def tune_catboost(X_train, y_train, X_test, y_test):
    """Tune CatBoost with RandomizedSearchCV"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Tuning CatBoost with RandomizedSearchCV...")
    logger.info("="*60)
    
    # Parameter grid
    param_dist = {
        'iterations': [300, 500, 700, 1000],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128],
        'bagging_temperature': [0, 0.5, 1.0]
    }
    
    logger.info("Parameter grid:")
    logger.info(f"  - iterations: {param_dist['iterations']}")
    logger.info(f"  - depth: {param_dist['depth']}")
    logger.info(f"  - learning_rate: {param_dist['learning_rate']}")
    logger.info(f"  Total combinations: {np.prod([len(v) for v in param_dist.values()]):,}")
    logger.info(f"  Testing: 30 random combinations with 3-fold CV")
    
    # RandomizedSearchCV
    cat_base = CatBoostClassifier(
        verbose=0,
        thread_count=-1,
        random_state=42
    )
    
    random_search = RandomizedSearchCV(
        estimator=cat_base,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring=make_scorer(average_precision_score, needs_proba=True),
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    logger.info("\n⏳ Starting RandomizedSearchCV (this will take 30-40 minutes)...")
    random_search.fit(X_train, y_train)
    
    # Best model
    best_cat = random_search.best_estimator_
    y_pred = best_cat.predict(X_test)
    y_probs = best_cat.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auprc = average_precision_score(y_test, y_probs)
    
    logger.info("\n✅ CatBoost Tuning Complete!")
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best CV score: {random_search.best_score_:.4f}")
    logger.info(f"Test - Precision: {report['1']['precision']:.2%}, Recall: {report['1']['recall']:.2%}, AUPRC: {auprc:.4f}")
    
    return {
        'model': best_cat,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'auprc': auprc,
        'best_params': random_search.best_params_
    }


def compare_results(baseline_results, tuned_xgb, tuned_cat):
    """Compare baseline vs tuned models"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Comparing Results...")
    logger.info("="*60)
    
    comparison_data = []
    
    # Baseline results
    for name, results in baseline_results.items():
        comparison_data.append({
            'Model': name,
            'Type': 'Baseline',
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1'],
            'AUPRC': results['auprc']
        })
    
    # Tuned results
    comparison_data.append({
        'Model': 'XGBoost',
        'Type': 'Tuned',
        'Precision': tuned_xgb['precision'],
        'Recall': tuned_xgb['recall'],
        'F1-Score': tuned_xgb['f1'],
        'AUPRC': tuned_xgb['auprc']
    })
    
    comparison_data.append({
        'Model': 'CatBoost',
        'Type': 'Tuned',
        'Precision': tuned_cat['precision'],
        'Recall': tuned_cat['recall'],
        'F1-Score': tuned_cat['f1'],
        'AUPRC': tuned_cat['auprc']
    })
    
    df = pd.DataFrame(comparison_data)
    df.to_csv('results/tuning_comparison.csv', index=False)
    
    print("\n" + "="*60)
    print("BASELINE vs TUNED COMPARISON")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Calculate improvements
    xgb_baseline = baseline_results['XGBoost (Baseline)']
    cat_baseline = baseline_results['CatBoost (Baseline)']
    
    xgb_improvement = (tuned_xgb['precision'] - xgb_baseline['precision']) * 100
    cat_improvement = (tuned_cat['precision'] - cat_baseline['precision']) * 100
    
    print(f"\n📈 Improvements:")
    print(f"  XGBoost Precision: +{xgb_improvement:.2f}%")
    print(f"  CatBoost Precision: +{cat_improvement:.2f}%")
    
    return df


def save_best_model(baseline_results, tuned_xgb, tuned_cat):
    """Save the best model"""
    logger.info("\n" + "="*60)
    logger.info("Saving best model...")
    logger.info("="*60)
    
    # Compare all models
    all_models = {
        'XGBoost (Baseline)': baseline_results['XGBoost (Baseline)'],
        'CatBoost (Baseline)': baseline_results['CatBoost (Baseline)'],
        'XGBoost (Tuned)': tuned_xgb,
        'CatBoost (Tuned)': tuned_cat
    }
    
    # Find best by AUPRC
    best_name = max(all_models.keys(), key=lambda k: all_models[k]['auprc'])
    best_model_info = all_models[best_name]
    
    # Save
    joblib.dump(best_model_info['model'], 'models/best_model.pkl')
    
    metadata = {
        'model_name': best_name,
        'metrics': {
            'precision': best_model_info['precision'],
            'recall': best_model_info['recall'],
            'f1': best_model_info['f1'],
            'auprc': best_model_info['auprc']
        }
    }
    
    if 'Tuned' in best_name:
        metadata['best_params'] = best_model_info['best_params']
    
    joblib.dump(metadata, 'models/model_metadata.pkl')
    
    logger.info(f"✅ Best Model: {best_name}")
    logger.info(f"   AUPRC: {best_model_info['auprc']:.4f}")
    logger.info(f"   Precision: {best_model_info['precision']:.2%}")
    logger.info(f"   Recall: {best_model_info['recall']:.2%}")
    logger.info("✅ Saved: models/best_model.pkl")
    
    return best_name


def main():
    """Run hyperparameter tuning pipeline"""
    
    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING WITH RANDOMIZEDSEARCHCV")
    print("="*60)
    print("\n⏰ Expected time: 1-2 hours")
    print("💡 This optimizes XGBoost and CatBoost for best performance\n")
    
    try:
        # Load data
        X_train, y_train, X_test, y_test = load_preprocessed_data()
        
        # Apply SMOTE-Tomek
        X_train_res, y_train_res = apply_smote_tomek(X_train, y_train)
        
        # Step 1: Train baseline models
        baseline_results = train_baseline_models(X_train_res, y_train_res, X_test, y_test)
        
        # Step 2: Tune XGBoost
        tuned_xgb = tune_xgboost(X_train_res, y_train_res, X_test, y_test)
        
        # Step 3: Tune CatBoost
        tuned_cat = tune_catboost(X_train_res, y_train_res, X_test, y_test)
        
        # Step 4: Compare results
        compare_results(baseline_results, tuned_xgb, tuned_cat)
        
        # Step 5: Save best model
        best_name = save_best_model(baseline_results, tuned_xgb, tuned_cat)
        
        # Success message
        print("\n" + "="*60)
        print("  ✅ HYPERPARAMETER TUNING COMPLETED!")
        print("="*60)
        print("\n📁 Saved Files:")
        print("  ✅ models/best_model.pkl (tuned model)")
        print("  ✅ results/tuning_comparison.csv")
        print(f"\n🏆 Best Model: {best_name}")
        print("\n🚀 Next Step:")
        print("  Run: python app.py")
        print("  (Will use the tuned model automatically!)")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Tuning failed: {e}")
        raise


if __name__ == "__main__":
    main()