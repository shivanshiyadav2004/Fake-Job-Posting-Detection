import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.utils import download_dataset, init_directories, DATASET_PATH, MODELS_DIR, ASSETS_DIR
from src.preprocessing import TextPreprocessor, display_stats
from src.evaluate import generate_eda_plots, evaluate_model_performance, save_model_evaluation_plots

def run_training_pipeline(vectorizer_type: str = "tfidf", sample_data: bool = False):
    """Runs the entire training pipeline: Preprocessing, EDA, Feature Engineering, Training, Grid Search, Evaluation, and Serialization."""
    init_directories()
    
    # 1. Download & Load Dataset
    print("\n--- Step 1: Loading Dataset ---")
    dataset_file = download_dataset()
    raw_df = pd.read_csv(dataset_file)
    
    # 2. Text Preprocessing
    print("\n--- Step 2: Preprocessing Dataset ---")
    preprocessor = TextPreprocessor()
    df_processed, stats = preprocessor.preprocess_dataset(raw_df)
    display_stats(stats)
    
    if sample_data and len(df_processed) > 5000:
        print("Sampling 5,000 records to accelerate training run...")
        # Maintain class ratio using train_test_split
        _, df_sampled = train_test_split(
            df_processed, 
            test_size=5000, 
            stratify=df_processed['fraudulent'], 
            random_state=42
        )
        df_processed = df_sampled.copy()
        print(f"Sampled class counts: {df_processed['fraudulent'].value_counts().to_dict()}")
        
    # Generate EDA plots
    print("\n--- Step 3: Generating EDA Visualization Plots ---")
    generate_eda_plots(raw_df, df_processed, ASSETS_DIR)
    
    # 3. Train-Test Split
    X = df_processed['cleaned_text']
    y = df_processed['fraudulent']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train split size: {len(X_train)} | Test split size: {len(X_test)}")
    
    # 4. Feature Engineering
    print(f"\n--- Step 4: Feature Extraction ({vectorizer_type.upper()}) ---")
    if vectorizer_type.lower() == "tfidf":
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    elif vectorizer_type.lower() == "count":
        vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    else:
        raise ValueError(f"Invalid vectorizer type: {vectorizer_type}")
        
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Define ML Models and GridSearchCV Parameter Grids
    print("\n--- Step 5: Setting up Model Configs & GridSearchCV Hyperparameters ---")
    
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "params": {
                "C": [0.1, 1.0, 10.0]
            }
        },
        "Multinomial Naive Bayes": {
            "model": MultinomialNB(),
            "params": {
                "alpha": [0.01, 0.1, 1.0]
            }
        },
        "Linear SVM": {
            # SGDClassifier with modified_huber gives probability estimates which allows ROC-AUC calculation
            "model": SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=42),
            "params": {
                "alpha": [1e-4, 1e-3, 1e-2]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(class_weight='balanced', random_state=42),
            "params": {
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20]
            }
        }
    }
    
    if XGB_AVAILABLE:
        # Calculate scale_pos_weight for imbalance
        scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
        models_config["XGBoost"] = {
            "model": XGBClassifier(scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1]
            }
        }
    else:
        print("XGBoost is not installed/available. Skipping XGBoost model.")

    # 6. Model Training & Parameter Tuning
    print("\n--- Step 6: Starting Model Training with Cross-Validation ---")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    comparison_results = []
    trained_models = {}
    
    for model_name, cfg in models_config.items():
        print(f"\nTraining and tuning: {model_name}...")
        grid = GridSearchCV(
            cfg["model"], 
            cfg["params"], 
            cv=cv, 
            scoring='f1', # optimize for F1 score due to imbalance
            n_jobs=-1, 
            verbose=1
        )
        grid.fit(X_train_vec, y_train)
        
        best_model = grid.best_estimator_
        trained_models[model_name] = best_model
        
        # Predictions
        y_pred = best_model.predict(X_test_vec)
        
        # Probabilities
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test_vec)[:, 1]
        elif hasattr(best_model, "decision_function"):
            # Calibrate decision function to 0-1 for ROC-AUC
            df = best_model.decision_function(X_test_vec)
            y_prob = 1 / (1 + np.exp(-df))
        else:
            y_prob = None
            
        # Evaluation
        metrics = evaluate_model_performance(y_test, y_pred, y_prob)
        save_model_evaluation_plots(y_test, y_pred, y_prob, model_name, ASSETS_DIR)
        
        print(f"[{model_name}] Best Params: {grid.best_params_}")
        print(f"[{model_name}] Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")
        
        # Save summary statistics (excluding raw confusion matrix and report dict for JSON storage)
        summary_metrics = {
            "Model Name": model_name,
            "Best Parameters": str(grid.best_params_),
            "Accuracy": round(metrics['accuracy'], 4),
            "Precision": round(metrics['precision'], 4),
            "Recall": round(metrics['recall'], 4),
            "F1-Score": round(metrics['f1'], 4),
            "ROC-AUC": round(metrics['roc_auc'], 4)
        }
        comparison_results.append(summary_metrics)
        
    # 7. Model Comparison Table and Best Model Selection
    comparison_df = pd.DataFrame(comparison_results)
    
    # Identify the best model based on F1-Score (since fake jobs detection is highly imbalanced)
    best_model_idx = comparison_df['F1-Score'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model Name']
    best_model_score = comparison_df.loc[best_model_idx, 'F1-Score']
    best_model_obj = trained_models[best_model_name]
    
    print("\n" + "=" * 60)
    print("                      MODEL COMPARISON SUMMARY              ")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    print(f"Identified BEST Model: {best_model_name} with F1-Score: {best_model_score:.4f}")
    print("=" * 60)
    
    # 8. Model Persistence
    print("\n--- Step 7: Saving Best Model and Vectorizer ---")
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
    comparison_results_path = os.path.join(MODELS_DIR, "model_comparison.json")
    vectorizer_meta_path = os.path.join(MODELS_DIR, "vectorizer_meta.json")
    
    joblib.dump(best_model_obj, best_model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save comparison table
    with open(comparison_results_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
        
    # Save vectorizer type metadata
    with open(vectorizer_meta_path, 'w') as f:
        json.dump({
            "vectorizer_type": vectorizer_type,
            "best_model_name": best_model_name
        }, f, indent=4)
        
    print(f"Saved best model object: {best_model_path}")
    print(f"Saved vectorizer object: {vectorizer_path}")
    print(f"Saved comparison logs: {comparison_results_path}")
    print(f"Saved vectorizer meta: {vectorizer_meta_path}")
    print("\nTraining Pipeline Completed successfully!")
    
    return best_model_name, comparison_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fake Job Posting Detection Training Pipeline")
    parser.add_argument("--vectorizer", type=str, default="tfidf", choices=["tfidf", "count"], help="Feature extraction vectorizer type")
    parser.add_argument("--sample", action="store_true", help="Sample dataset to speed up training")
    args = parser.parse_args()
    
    run_training_pipeline(vectorizer_type=args.vectorizer, sample_data=args.sample)
