# 🚀 Fake Job Posting Detection System

An end-to-end Machine Learning and Natural Language Processing (NLP) application that detects whether a job posting is legitimate or fraudulent.

The project leverages advanced text preprocessing, feature engineering, multiple machine learning algorithms, hyperparameter tuning, explainable AI techniques, and an interactive Streamlit dashboard to provide real-time fraud detection.

---

## 🌐 Live Demo

🚀 Try the Application:

https://fake-job-posting-detection-dasj6rqclcrdqdcyntottw.streamlit.app/

---

## 📌 Project Overview

Fake job postings have become increasingly common across online job portals, leading to financial loss and privacy risks for job seekers.

This project aims to solve that problem by building an intelligent system capable of identifying fraudulent job advertisements using Machine Learning and NLP techniques.

The application allows users to:

- Analyze job descriptions in real time
- Detect fraudulent postings
- Understand prediction reasoning
- Generate PDF reports
- Export prediction history
- Explore detailed dataset analytics

---

## ✨ Features

### 🔹 Data Preprocessing & Cleaning

- Text normalization
- Lowercase conversion
- URL removal
- Email removal
- Punctuation removal
- Number removal
- Stopword removal
- Lemmatization using NLTK WordNet

### 🔹 Exploratory Data Analysis (EDA)

- Missing Value Analysis
- Fraud vs Legitimate Distribution
- Word Frequency Analysis
- Word Clouds
- Location Distribution
- Employment Type Analysis
- Metadata Correlation Analysis

### 🔹 Machine Learning Models

The system automatically trains and evaluates multiple models:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Decision Tree
- Random Forest
- XGBoost

### 🔹 Hyperparameter Optimization

- GridSearchCV
- Cross Validation
- F1-Score Optimization
- Model Comparison

### 🔹 Explainable AI

Provides keyword-based explanations showing which terms contributed most to the prediction.

### 🔹 Reporting & Auditing

- PDF Report Generation
- CSV Export
- Prediction History Tracking

### 🔹 Interactive Streamlit Dashboard

- Dark Mode Interface
- Real-Time Predictions
- Model Retraining
- EDA Gallery
- Sample Job Templates

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-Learn
- XGBoost

### NLP

- NLTK
- TF-IDF Vectorization

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn
- WordCloud

### Web Framework

- Streamlit

### Model Persistence

- Joblib

---

## 📂 Project Structure

```text
Fake_JOB_Posting_Portal/
│
├── assets/
│   └── Images and visual assets
│
├── dataset/
│   ├── fake_job_postings.csv
│   └── prediction_history.csv
│
├── models/
│   ├── best_model.joblib
│   ├── vectorizer.joblib
│   ├── model_comparison.json
│   └── vectorizer_meta.json
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── prediction.py
│   ├── explainability.py
│   └── visualization.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Machine Learning Pipeline

### Step 1: Data Collection

- Import dataset
- Validate records

### Step 2: Data Preprocessing

- Clean text
- Remove noise
- Lemmatization
- Stopword filtering

### Step 3: Feature Engineering

- TF-IDF Vectorization

### Step 4: Model Training

- Train multiple algorithms
- Compare performance

### Step 5: Hyperparameter Tuning

- GridSearchCV Optimization

### Step 6: Evaluation

- Accuracy
- Precision
- Recall
- F1-Score

### Step 7: Deployment

- Streamlit Web Application

---

## 📊 Key Functionalities

### Fraud Detection

Predicts whether a job posting is:

✅ Legitimate

❌ Fraudulent

### Explainable Predictions

Displays important keywords influencing the prediction.

### Model Retraining

Allows users to retrain models directly from the dashboard.

### Prediction History

Stores all predictions for auditing and review.

### Report Generation

Generates downloadable PDF reports.

---

## 📸 Screenshots

### Home Page

![Home](./assets/home.png)

### Prediction Dashboard

![Dashboard](./assets/dashboard.png)

### Login Page

![Login](./assets/login.png)

### Registration Page

![Register](./assets/register.png)

### Report Generation

![Report](./assets/report.png)

---

## 🎯 Project Highlights

✔ Built an end-to-end NLP-based fraud detection system

✔ Applied Machine Learning and Natural Language Processing techniques

✔ Implemented multiple classification models

✔ Performed hyperparameter optimization using GridSearchCV

✔ Developed an interactive Streamlit dashboard

✔ Added explainable AI features

✔ Integrated PDF report generation

✔ Enabled prediction history tracking and export

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/shivanshiyadav2004/Fake_JOB_Posting_Portal.git
```

### Navigate to Project

```bash
cd Fake_JOB_Posting_Portal
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

- Deep Learning Models (LSTM/BERT)
- Real-Time API Integration
- Cloud Deployment
- User Authentication
- Advanced Explainable AI
- Multilingual Fraud Detection

---

## 👨‍💻 Author

**Shivanshi Yadav**

Computer Science Engineering Student

- GitHub: https://github.com/shivanshiyadav2004
- LinkedIn: https://www.linkedin.com/in/shivanshi-yadav-593188308

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
