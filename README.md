# Fake Job Posting Detection System

An end-to-end Machine Learning and Natural Language Processing (NLP) web application that detects whether job postings are legitimate or fraudulent. The application features multi-model training with hyperparameter grid search, detailed Exploratory Data Analysis (EDA) visualizations, real-time predictions, local prediction explainability, prediction history logging, and custom PDF report exports.

---

## 🚀 Features
- **Data Preprocessing & Cleaning**: Automates text conversion to lowercase, URL and email removal, punctuation/number stripping, stopword filtering, and lemmatization (via NLTK WordNet).
- **Exploratory Data Analysis (EDA)**: Programmatically generates and persists interactive charts (missing values heatmaps, class distributions, word frequencies, word clouds, location frequencies, employment types, and metadata correlations).
- **Multi-model Training & Tuning**: Automatically benchmarks:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM (SGDClassifier with modified huber loss)
  - Decision Trees
  - Random Forests
  - XGBoost (if package is compiled)
  - Uses `GridSearchCV` to optimize the best F1-Score to handle class imbalances.
- **Explainable Predictions**: Extracts keyword influences for predictions, highlighting which words push the model towards categorizing a listing as legitimate or fraudulent.
- **Reporting & Auditing**:
  - Download prediction reports as beautifully formatted PDFs.
  - Export prediction history as a CSV file.
- **Interactive Streamlit Web Dashboard**: Sleek dark mode UI with card layouts, quick mock templates for testing, real-time prediction forms, EDA gallery, and interactive retrainer control.

---

## 📂 Project Structure

```
Fake_JOB_Posting_Portal/
│
├── dataset/                    # Stores datasets & prediction history
│   └── fake_job_postings.csv   # Target dataset (automatically downloaded)
│
├── models/                     # Saved models and vectorizers
│   ├── best_model.joblib       # Persisted top-performing model
│   ├── vectorizer.joblib       # Persisted TF-IDF/Count vectorizer
│   ├── model_comparison.json   # Model training comparison statistics
│   └── vectorizer_meta.json    # Vectorizer configurations
│
├── src/                        # Source python scripts
│   ├── preprocessing.py        # Text cleaning and lemmatization
│   ├── train.py                # Hyperparameter tuning and model training
│   ├── evaluate.py             # EDA plotting and performance metric evaluations
│   ├── predict.py              # Real-time predictor and keyword explainers
│   └── utils.py                # Folder, download, PDF, and history utilities
│
├── assets/                     # Saved EDA and model performance charts
├── app.py                      # Streamlit application web entry point
├── requirements.txt            # Python dependencies list
└── README.md                   # Project documentation
```

---

## 🛠️ Installation Guide

### Prerequisites
- Python 3.11 or later
- Access to terminal/command prompt

### Setup Steps
1. **Clone or Navigate to Project Directory**:
   ```bash
   cd c:/Users/suraj/OneDrive/Desktop/Fake_JOB_Posting_Portal
   ```

2. **Create a Virtual Environment (Recommended)**:
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage & Running

### 1. Run Model Training (First-time setup)
To initialize folders, download the Kaggle dataset (~50MB), generate EDA plots, perform grid search hyperparameter tuning across all models, and persist the best model, execute the train script:
```bash
# Recommended: Run on the full dataset
python src/train.py

# Optional: Run on a sampled dataset (5,000 records) to speed up Grid Search locally
python src/train.py --sample
```

You can choose to use CountVectorizer instead of TF-IDF using the `--vectorizer` flag:
```bash
python src/train.py --vectorizer count
```

### 2. Run the Streamlit Web Application
Once training completes, start the web interface:
```bash
streamlit run app.py
```
This will spin up a local server. Your web browser should automatically open the dashboard page.

---

## 🏗️ Architecture & Processing Pipeline

```
+--------------------------------------------------------+
|                      Streamlit UI                      |
| (Overview stats, EDA plots, prediction, retraining)    |
+---------------------------+----------------------------+
                            |
                     predicts input text
                            |
                            v
+--------------------------------------------------------+
|                   NLP Preprocessing                    |
| (Text Fusion -> Regex Cleaning -> Stopwords -> Lemma) |
+---------------------------+----------------------------+
                            |
                     vectorized words
                            |
                            v
+--------------------------------------------------------+
|             Vectorizer & Model Persistence             |
|   (loads joblib vectorizers & tuned classifier)        |
+---------------------------+----------------------------+
                            |
                    predictions & proba
                            |
                            v
+--------------------------------------------------------+
|                     PDF Generator                      |
|     (Outputs styled visual verification report)        |
+--------------------------------------------------------+
```

1. **Preprocessing Pipeline**: Combines text from the `title`, `company_profile`, `description`, `requirements`, `benefits`, and `location` columns. The text is lowercased, and punctuation, numbers, special characters, and NLTK stop words are removed. Words are lemmatized to their dictionary base form.
2. **Feature Extraction**: Engineered using TF-IDF Vectorization or CountVectorizer (Bag of Words) with unigram and bigram representation (`ngram_range=(1,2)`) and a limit of 5,000 top features.
3. **Model Selection**: Optimizes F1-Score via Grid Search across classifiers. The best estimator is serialized as `models/best_model.joblib`.
4. **Predictive Analytics**: Inputs from the prediction form are tokenized and scored. If the model is linear, exact coefficients are used to present local keyword explanations. If the model is tree-based, global importances scaled by TF-IDF scores are used.
