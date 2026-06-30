import re
import string
import nltk
import pandas as pd
import numpy as np

# Downloader function for NLTK resources
def download_nltk_resources():
    resources = {
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'punkt': 'tokenizers/punkt'
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

class TextPreprocessor:
    """Class to handle cleaning and preprocessing of the Fake Job Postings dataset."""
    
    def __init__(self):
        download_nltk_resources()
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add common email/web garbage words if any
        self.stop_words.update(['us', 'would', 'get', 'like', 'one', 'xml', 'amp', 'nbsp'])

    def clean_text(self, text: str) -> str:
        """Cleans and processes a single text string."""
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 3. Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Remove special characters/symbols and punctuation
        # Replace punctuation with whitespace to prevent joining words
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        # 5. Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # 6. Replace multiple spaces and newlines with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 7. Tokenize, remove stopwords, and lemmatize
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 2 # remove very short tokens
        ]
        
        return " ".join(cleaned_words)

    def preprocess_dataset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Preprocesses the entire dataframe, handles duplicates/nulls, combines features, and returns stats."""
        stats = {}
        
        # Total initial rows
        stats['initial_records'] = len(df)
        
        # Missing values count per column
        text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits', 'location']
        stats['missing_values'] = df[text_cols].isnull().sum().to_dict()
        
        # Fill missing values with empty strings
        for col in text_cols:
            df[col] = df[col].fillna('')
            
        # Combine text fields into a single text feature
        df['combined_text'] = (
            df['title'] + " " + 
            df['company_profile'] + " " + 
            df['description'] + " " + 
            df['requirements'] + " " + 
            df['benefits'] + " " + 
            df['location']
        )
        
        # Save average text lengths before cleaning
        stats['avg_word_count_raw'] = df['combined_text'].apply(lambda x: len(x.split())).mean()
        
        # Remove duplicate records
        df_dedup = df.drop_duplicates(subset=['combined_text', 'fraudulent']).copy()
        stats['duplicates_removed'] = len(df) - len(df_dedup)
        stats['final_records'] = len(df_dedup)
        
        # Apply text cleaning
        print("Cleaning text data... (this might take a minute)")
        df_dedup['cleaned_text'] = df_dedup['combined_text'].apply(self.clean_text)
        
        # Remove any rows where cleaned_text is empty
        df_dedup = df_dedup[df_dedup['cleaned_text'] != '']
        stats['empty_records_removed'] = stats['final_records'] - len(df_dedup)
        stats['final_records'] = len(df_dedup)
        
        # Save average text lengths after cleaning
        stats['avg_word_count_cleaned'] = df_dedup['cleaned_text'].apply(lambda x: len(x.split())).mean()
        
        # Class distribution
        class_counts = df_dedup['fraudulent'].value_counts().to_dict()
        stats['legitimate_count'] = class_counts.get(0, 0)
        stats['fraudulent_count'] = class_counts.get(1, 0)
        stats['fraud_percentage'] = (stats['fraudulent_count'] / stats['final_records']) * 100
        
        return df_dedup, stats

def display_stats(stats: dict):
    """Prints preprocessing statistics to console."""
    print("=" * 50)
    print("           DATA PREPROCESSING STATISTICS           ")
    print("=" * 50)
    print(f"Initial Records:         {stats['initial_records']}")
    print(f"Duplicates Removed:      {stats['duplicates_removed']}")
    print(f"Empty Cleaned Removed:   {stats['empty_records_removed']}")
    print(f"Final Cleaned Records:   {stats['final_records']}")
    print("-" * 50)
    print(f"Legitimate Jobs (0):     {stats['legitimate_count']}")
    print(f"Fraudulent Jobs (1):     {stats['fraudulent_count']}")
    print(f"Fraud Percentage:        {stats['fraud_percentage']:.2f}%")
    print("-" * 50)
    print(f"Avg Raw Word Count:      {stats['avg_word_count_raw']:.2f}")
    print(f"Avg Cleaned Word Count:  {stats['avg_word_count_cleaned']:.2f}")
    print("=" * 50)
