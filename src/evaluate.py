import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from collections import Counter

# Set custom styling for plots (matching dark theme where appropriate, or clean modern look)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.facecolor': '#181c24',
    'axes.facecolor': '#181c24',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#cbd5e1',
    'ytick.color': '#cbd5e1',
    'axes.edgecolor': '#475569',
    'grid.color': '#334155'
})

def generate_eda_plots(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, assets_dir: str):
    """Generates all requested EDA plots and saves them in the assets folder."""
    os.makedirs(assets_dir, exist_ok=True)
    print("Generating EDA visualizations...")

    # 1. Missing Value Heatmap (on raw dataframe before imputation)
    plt.figure(figsize=(10, 6))
    # Select columns of interest for readability
    cols_to_check = ['title', 'location', 'department', 'salary_range', 
                     'company_profile', 'description', 'requirements', 
                     'benefits', 'employment_type', 'required_experience', 
                     'required_education', 'industry', 'function']
    sns.heatmap(raw_df[cols_to_check].isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Value Heatmap (Raw Data)", fontsize=14, color='#e2e8f0')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "missing_value_heatmap.png"), dpi=150, facecolor='#181c24')
    plt.close()

    # 2. Class Distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='fraudulent', data=cleaned_df, palette=['#3b82f6', '#ef4444'])
    plt.title("Class Distribution (Legitimate vs Fraudulent)", fontsize=14, color='#e2e8f0')
    plt.xlabel("Class (0: Legitimate, 1: Fraudulent)", color='#e2e8f0')
    plt.ylabel("Count", color='#e2e8f0')
    # Add count values on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='#cbd5e1')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "class_distribution.png"), dpi=150, facecolor='#181c24')
    plt.close()

    # 3. Top Hiring Locations
    plt.figure(figsize=(10, 6))
    # Standard format is "US, NY, New York" or just country code, let's extract first non-empty code
    locations = cleaned_df['location'].apply(lambda x: x.split(',')[0].strip() if ',' in x else x.strip())
    locations = locations[locations != '']
    top_locs = locations.value_counts().head(10)
    sns.barplot(x=top_locs.values, y=top_locs.index, palette='Blues_r')
    plt.title("Top 10 Hiring Countries/Locations", fontsize=14, color='#e2e8f0')
    plt.xlabel("Number of Postings", color='#e2e8f0')
    plt.ylabel("Country Code", color='#e2e8f0')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "top_locations.png"), dpi=150, facecolor='#181c24')
    plt.close()

    # 4. Employment Type Distribution
    plt.figure(figsize=(8, 5))
    emp_types = cleaned_df['employment_type'].replace('', 'Not Specified')
    emp_counts = emp_types.value_counts()
    sns.barplot(x=emp_counts.values, y=emp_counts.index, palette='crest')
    plt.title("Employment Type Distribution", fontsize=14, color='#e2e8f0')
    plt.xlabel("Count", color='#e2e8f0')
    plt.ylabel("Employment Type", color='#e2e8f0')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "employment_types.png"), dpi=150, facecolor='#181c24')
    plt.close()

    # 5. Correlation Heatmap of Meta Columns
    plt.figure(figsize=(8, 6))
    meta_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']
    # Check if these columns exist in the raw dataframe
    existing_meta = [c for c in meta_cols if c in raw_df.columns]
    if len(existing_meta) > 1:
        corr = raw_df[existing_meta].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".4f", vmin=-1, vmax=1, square=True)
        plt.title("Correlation Heatmap of Metadata Features", fontsize=14, color='#e2e8f0')
        plt.tight_layout()
        plt.savefig(os.path.join(assets_dir, "correlation_heatmap.png"), dpi=150, facecolor='#181c24')
    plt.close()

    # 6. Word Frequency
    all_words = " ".join(cleaned_df['cleaned_text']).split()
    word_counts = Counter(all_words)
    top_words = pd.DataFrame(word_counts.most_common(20), columns=['word', 'count'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='word', data=top_words, palette='viridis')
    plt.title("Top 20 Most Frequent Words (Cleaned Text)", fontsize=14, color='#e2e8f0')
    plt.xlabel("Frequency", color='#e2e8f0')
    plt.ylabel("Words", color='#e2e8f0')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "word_frequency.png"), dpi=150, facecolor='#181c24')
    plt.close()

    # 7. WordCloud for Legitimate Jobs
    legit_text = " ".join(cleaned_df[cleaned_df['fraudulent'] == 0]['cleaned_text'])
    if legit_text:
        wordcloud_legit = WordCloud(
            width=800, height=400, background_color='#181c24', 
            colormap='GnBu', max_words=100
        ).generate(legit_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_legit, interpolation='bilinear')
        plt.axis('off')
        plt.title("WordCloud: Legitimate Job Postings", fontsize=16, color='#e2e8f0', pad=15)
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(assets_dir, "wordcloud_legitimate.png"), dpi=150, facecolor='#181c24')
        plt.close()

    # 8. WordCloud for Fraudulent Jobs
    fraud_text = " ".join(cleaned_df[cleaned_df['fraudulent'] == 1]['cleaned_text'])
    if fraud_text:
        wordcloud_fraud = WordCloud(
            width=800, height=400, background_color='#181c24', 
            colormap='OrRd', max_words=100
        ).generate(fraud_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fraud, interpolation='bilinear')
        plt.axis('off')
        plt.title("WordCloud: Fraudulent Job Postings", fontsize=16, color='#e2e8f0', pad=15)
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(assets_dir, "wordcloud_fraudulent.png"), dpi=150, facecolor='#181c24')
        plt.close()
        
    print("EDA visualizations saved in assets folder.")

def evaluate_model_performance(y_true, y_pred, y_prob=None) -> dict:
    """Computes all standard performance evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    return metrics

def save_model_evaluation_plots(y_true, y_pred, y_prob, model_name: str, assets_dir: str):
    """Saves Confusion Matrix and ROC Curve plots for a specific trained model."""
    os.makedirs(assets_dir, exist_ok=True)
    safe_name = model_name.replace(" ", "_").lower()
    
    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], 
                yticklabels=['Legit', 'Fraud'],
                cbar=False, annot_kws={"size": 14})
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, color='#e2e8f0', pad=10)
    plt.ylabel('Actual Category', color='#e2e8f0', labelpad=10)
    plt.xlabel('Predicted Category', color='#e2e8f0', labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, f"cm_{safe_name}.png"), dpi=150, facecolor='#181c24')
    plt.close()

    # 2. ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='#3b82f6', lw=2.5, label=f'ROC Curve (AUC = {auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='#ef4444', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(f"ROC Curve: {model_name}", fontsize=14, color='#e2e8f0', pad=10)
        plt.xlabel('False Positive Rate', color='#e2e8f0')
        plt.ylabel('True Positive Rate', color='#e2e8f0')
        plt.legend(loc="lower right", facecolor='#181c24', edgecolor='#cbd5e1')
        plt.tight_layout()
        plt.savefig(os.path.join(assets_dir, f"roc_{safe_name}.png"), dpi=150, facecolor='#181c24')
        plt.close()
