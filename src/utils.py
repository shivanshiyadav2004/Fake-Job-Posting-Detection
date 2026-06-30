import os
import urllib.request
import pandas as pd
from datetime import datetime
from fpdf import FPDF

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "fake_job_postings.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
HISTORY_PATH = os.path.join(DATASET_DIR, "prediction_history.csv")

# Mirror URLs for the dataset
DATASET_URLS = [
    "https://raw.githubusercontent.com/Cindyalifia/bangkit-project-1/master/fake_job_postings.csv",
    "https://raw.githubusercontent.com/38832/Fake-Job-Posting-Prediction/master/fake_job_postings.csv",
    "https://raw.githubusercontent.com/38832/Fake-Job-Posting-Prediction/main/fake_job_postings.csv"
]

def init_directories():
    """Initializes the project directory structure."""
    for folder in [DATASET_DIR, MODELS_DIR, ASSETS_DIR]:
        os.makedirs(folder, exist_ok=True)
    print("Project directories verified.")

def download_dataset():
    """Downloads the Fake Job Postings dataset from mirrors if not present locally."""
    init_directories()
    
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at: {DATASET_PATH}")
        return DATASET_PATH
        
    print("Dataset not found locally. Initiating download...")
    
    for url in DATASET_URLS:
        try:
            print(f"Attempting to download from: {url}")
            # Use urllib to download to avoid external dependency issues before pip install
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req) as response, open(DATASET_PATH, 'wb') as out_file:
                # Use chunked copying to show progress if possible
                meta = response.info()
                file_size = int(meta.get("Content-Length", 0))
                print(f"File size: {file_size / (1024*1024):.2f} MB")
                
                block_sz = 8192
                downloaded = 0
                while True:
                    buffer = response.read(block_sz)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if file_size:
                        percent = downloaded * 100.0 / file_size
                        print(f"Downloaded {percent:.2f}% ({downloaded / (1024*1024):.2f} MB)", end="\r")
                print("\nDownload complete!")
            return DATASET_PATH
        except Exception as e:
            print(f"Failed to download from {url}. Error: {e}")
            if os.path.exists(DATASET_PATH):
                os.remove(DATASET_PATH) # Remove partial file on error
            continue
            
    raise FileNotFoundError("Could not download dataset from any available mirrors. Please place fake_job_postings.csv inside the 'dataset/' folder.")

def save_prediction_to_history(job_details: dict, prediction: str, confidence: float):
    """Saves a prediction entry to the CSV history log."""
    init_directories()
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": job_details.get("title", ""),
        "location": job_details.get("location", ""),
        "company_profile": job_details.get("company_profile", "")[:100] + "...", # truncate for history file
        "description": job_details.get("description", "")[:100] + "...",
        "requirements": job_details.get("requirements", "")[:100] + "...",
        "benefits": job_details.get("benefits", "")[:100] + "...",
        "prediction": prediction,
        "confidence": f"{confidence:.2%}"
    }
    df = pd.DataFrame([row])
    
    if not os.path.exists(HISTORY_PATH):
        df.to_csv(HISTORY_PATH, index=False)
    else:
        df.to_csv(HISTORY_PATH, mode='a', header=False, index=False)

def get_prediction_history():
    """Reads and returns the prediction history log."""
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=["timestamp", "title", "location", "company_profile", "description", "requirements", "benefits", "prediction", "confidence"])
    try:
        return pd.read_csv(HISTORY_PATH)
    except Exception as e:
        print(f"Error reading history file: {e}")
        return pd.DataFrame()

class PDFReport(FPDF):
    def header(self):
        # Draw header
        self.set_fill_color(24, 28, 36) # dark background color matching dark theme
        self.rect(0, 0, 210, 40, 'F')
        
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 18)
        self.cell(0, 10, 'FAKE JOB POSTING DETECTION REPORT', border=0, ln=1, align='C')
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 10, 'Automated NLP & Machine Learning Verification System', border=0, ln=1, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Generated by Antigravity AI Job Verifier', border=0, ln=0, align='C')

def generate_pdf_report(job_details: dict, prediction: str, confidence: float, important_keywords: list, output_pdf_path: str):
    """Generates a clean, professional PDF report of the verification results."""
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Spacer
    pdf.ln(5)
    
    # Metadata Box
    pdf.set_fill_color(245, 246, 248)
    pdf.set_draw_color(220, 224, 230)
    pdf.rect(10, 45, 190, 25, 'FD')
    
    pdf.set_y(47)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(10)
    pdf.cell(50, 6, "Job Title:", ln=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(130, 6, str(job_details.get("title", "N/A")), ln=1)
    
    pdf.cell(10)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(50, 6, "Location:", ln=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(130, 6, str(job_details.get("location", "N/A")), ln=1)

    pdf.cell(10)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(50, 6, "Analysis Date:", ln=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(130, 6, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=1)
    
    pdf.ln(15)
    
    # Verification Result (Big Status Bar)
    is_fraud = prediction.lower() == "fraudulent"
    if is_fraud:
        pdf.set_fill_color(254, 226, 226) # light red
        pdf.set_draw_color(239, 68, 68)   # red border
        pdf.set_text_color(185, 28, 28)   # dark red text
        status_text = "WARNING: FRAUDULENT JOB POSTING DETECTED"
    else:
        pdf.set_fill_color(240, 253, 244) # light green
        pdf.set_draw_color(34, 197, 94)   # green border
        pdf.set_text_color(21, 128, 61)   # dark green text
        status_text = "VERIFIED: LEGITIMATE JOB POSTING"
        
    pdf.rect(10, 78, 190, 18, 'FD')
    pdf.set_y(83)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 8, status_text, align='C', ln=1)
    
    # Confidence Score Box
    pdf.ln(5)
    pdf.set_text_color(80, 80, 80)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f"Prediction Confidence: {confidence:.2%}", align='C', ln=1)
    
    pdf.ln(8)
    
    # Job Details Breakdown
    pdf.set_text_color(24, 28, 36)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, "Submitted Job Details Summary", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    sections = [
        ("Company Profile", "company_profile"),
        ("Job Description", "description"),
        ("Requirements", "requirements"),
        ("Benefits", "benefits")
    ]
    
    pdf.set_font('Helvetica', '', 10)
    for title, key in sections:
        val = job_details.get(key, "").strip()
        if val:
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 6, title, ln=1)
            pdf.set_font('Helvetica', '', 10)
            # Safe multi-line write
            pdf.multi_cell(0, 5, val[:500] + ("..." if len(val) > 500 else ""), border=0)
            pdf.ln(4)
            
    # Key Predictors / Important Keywords
    if important_keywords:
        pdf.ln(4)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, "Key NLP Predictors Influencing the Result", ln=1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
        pdf.set_font('Helvetica', '', 10)
        kw_text = ", ".join([f"{word} ({weight:+.4f})" for word, weight in important_keywords[:10]])
        pdf.multi_cell(0, 5, f"The following keywords present in the text had the highest statistical influence on the classification model:\n{kw_text}")
        pdf.ln(5)
        
    # Standard warning disclaimer
    pdf.ln(10)
    pdf.set_fill_color(249, 250, 251)
    pdf.rect(10, pdf.get_y(), 190, 20, 'F')
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, "Disclaimer: This verification report is generated by a Machine Learning model trained on historical data. While it holds high accuracy, it should be used as a support tool alongside manual due diligence. Do not share sensitive personal information or pay any fees for job applications.", align='C')

    pdf.output(output_pdf_path)
    print(f"Report saved to: {output_pdf_path}")
