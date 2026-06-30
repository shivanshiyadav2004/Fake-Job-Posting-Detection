import os
import json
import joblib
import numpy as np
from src.preprocessing import TextPreprocessor
from src.utils import MODELS_DIR

class JobPredictor:
    """Class to handle loading the best model and predicting whether a job posting is legitimate or fraudulent."""
    
    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        self.vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
        self.meta_path = os.path.join(MODELS_DIR, "vectorizer_meta.json")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError("Model files not found. Please train models first using src/train.py.")
            
        # Load persisted assets
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        
        # Load metadata
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
            
        self.vectorizer_type = self.meta.get("vectorizer_type", "tfidf")
        self.model_name = self.meta.get("best_model_name", "Unknown Model")
        
        # Initialize preprocessing clean logic
        self.preprocessor = TextPreprocessor()

    def predict(self, job_details: dict) -> tuple[str, float, list]:
        """
        Predicts if a job posting is legitimate or fraudulent.
        
        Parameters:
        job_details (dict): Keys should include: title, company_profile, description, requirements, benefits, location
        
        Returns:
        prediction (str): "Legitimate" or "Fraudulent"
        confidence (float): Probability score of the predicted class
        important_keywords (list): Top keywords present in the text and their weights
        """
        # 1. Combine fields
        combined_text = (
            f"{job_details.get('title', '')} "
            f"{job_details.get('company_profile', '')} "
            f"{job_details.get('description', '')} "
            f"{job_details.get('requirements', '')} "
            f"{job_details.get('benefits', '')} "
            f"{job_details.get('location', '')}"
        )
        
        # 2. Preprocess text
        cleaned_text = self.preprocessor.clean_text(combined_text)
        
        if not cleaned_text:
            return "Legitimate", 0.50, [] # Default fallback
            
        # 3. Vectorize text
        vectorized_text = self.vectorizer.transform([cleaned_text])
        
        # 4. Predict
        pred_class = int(self.model.predict(vectorized_text)[0])
        
        # 5. Extract probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vectorized_text)[0]
            confidence = probs[pred_class]
            fraud_prob = probs[1]
        else:
            # Calibrate SVM decision function output if decision function exists
            if hasattr(self.model, "decision_function"):
                df_val = self.model.decision_function(vectorized_text)[0]
                fraud_prob = 1 / (1 + np.exp(-df_val))
                confidence = fraud_prob if pred_class == 1 else (1 - fraud_prob)
            else:
                confidence = 1.0
                fraud_prob = float(pred_class)
                
        prediction_label = "Fraudulent" if pred_class == 1 else "Legitimate"
        
        # 6. Extract important keywords influencing the prediction
        important_keywords = self._explain_prediction(cleaned_text, vectorized_text, pred_class)
        
        return prediction_label, float(confidence), important_keywords

    def _explain_prediction(self, cleaned_text: str, vectorized_text, pred_class: int) -> list:
        """Helper to compute feature importances locally for words in the input text."""
        words_in_input = list(set(cleaned_text.split()))
        feature_names = self.vectorizer.get_feature_names_out()
        vocab = self.vectorizer.vocabulary_
        
        # Retrieve non-zero components of the tf-idf/count vector
        vectorized_array = vectorized_text.toarray()[0]
        
        word_weights = []
        
        # Linear models (Logistic Regression, Linear SVM)
        if hasattr(self.model, "coef_"):
            coefficients = self.model.coef_[0]
            for word in words_in_input:
                if word in vocab:
                    idx = vocab[word]
                    tf_val = vectorized_array[idx]
                    if tf_val > 0:
                        weight = coefficients[idx]
                        # Score indicates contribution to fraud (positive) vs legit (negative)
                        word_weights.append((word, float(weight)))
                        
        # Naive Bayes
        elif hasattr(self.model, "feature_log_prob_"):
            # MultinomialNB log prob differences
            log_prob_diff = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]
            for word in words_in_input:
                if word in vocab:
                    idx = vocab[word]
                    tf_val = vectorized_array[idx]
                    if tf_val > 0:
                        weight = log_prob_diff[idx]
                        word_weights.append((word, float(weight)))
                        
        # Tree-based models (Random Forest, Decision Tree, XGBoost)
        elif hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            # Since global importances are positive only, we show the top TF-IDF * importance words in input
            for word in words_in_input:
                if word in vocab:
                    idx = vocab[word]
                    tf_val = vectorized_array[idx]
                    if tf_val > 0:
                        weight = importances[idx] * tf_val
                        # Sign it based on class prediction to keep visualization consistent
                        # (higher weight means highly important for prediction)
                        sign = 1.0 if pred_class == 1 else -1.0
                        word_weights.append((word, float(weight * sign)))
        
        # If no weights could be calculated, return empty list
        if not word_weights:
            return []
            
        # Sort based on absolute value of weights (highest influence first)
        word_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        return word_weights[:15] # Return top 15 most important keywords
