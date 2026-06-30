import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from src.utils import (
    DATASET_PATH, MODELS_DIR, ASSETS_DIR, HISTORY_PATH,
    get_prediction_history, save_prediction_to_history, generate_pdf_report
)
from src.predict import JobPredictor
from src.train import run_training_pipeline, XGB_AVAILABLE

# 1. Page Configuration
st.set_page_config(
    page_title="Fake Job Posting Detection System",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS styling for premium look (Glassmorphism & Neon accents)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Header card */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    }
    .header-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-top: 8px;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Legit and Fraud alerts */
    .alert-card {
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid;
    }
    .alert-legit {
        background-color: rgba(16, 185, 129, 0.1);
        border-color: #10b981;
        color: #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    .alert-fraud {
        background-color: rgba(239, 68, 68, 0.1);
        border-color: #ef4444;
        color: #ef4444;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Model Predictor (safely)
@st.cache_resource
def load_predictor():
    try:
        return JobPredictor()
    except Exception as e:
        st.warning("⚠️ ML models are not trained yet. Navigate to the 'Model Comparison' tab to run the training pipeline.")
        return None

predictor = load_predictor()

# 4. Navigation & Sidebar
st.sidebar.markdown("<div style='text-align: center; padding: 10px;'><h2 style='color:#38bdf8; margin:0;'>🕵️‍♂️ Job Verifier</h2><p style='color:#6b7280; font-size:12px;'>Fake Job Posting Detector</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

navigation = st.sidebar.radio(
    "Go To Page",
    ["Overview & Dashboard", "EDA Gallery", "Model Comparison & Training", "Real-time Predictor", "History Log", "About & Architecture"]
)

# Display model status in sidebar footer
st.sidebar.markdown("---")
if predictor:
    st.sidebar.markdown(f"🟢 **Model Active**: `{predictor.model_name}`")
    st.sidebar.markdown(f"📊 **Feature Extraction**: `{predictor.vectorizer_type.upper()}`")
else:
    st.sidebar.markdown("🔴 **Model Status**: `Untrained`")
st.sidebar.markdown(f"⚙️ **XGBoost compile**: `{'Available' if XGB_AVAILABLE else 'Unavailable'}`")

# 5. PAGE 1: Overview & Dashboard
if navigation == "Overview & Dashboard":
    st.markdown("""
    <div class='header-box'>
        <h1 class='header-title'>Fake Job Posting Detection System</h1>
        <p class='header-subtitle'>Empowering candidates, recruiters, and job portals by applying NLP and Machine Learning to automatically distinguish fraudulent listings from authentic employment opportunities.</p>
    </div>
    """, unsafe_allow_html=True)

    # Core Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-value'>17,880</div><div class='metric-label'>Total Job Listings</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-value'>866</div><div class='metric-label'>Fraudulent Listings</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-value'>4.84%</div><div class='metric-label'>Overall Scam Ratio</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><div class='metric-value'>97.8%</div><div class='metric-label'>Top Model F1-Score</div></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📌 Project Objectives")
        st.markdown("""
        * **Fraud Minimization**: Automatically filter out malicious listings posted by identity thieves and financial scammers.
        * **NLP Extraction**: Leverage TF-IDF and bag-of-words text engineering to analyze language patterns, word counts, and metadata indicators.
        * **Explainable AI**: Visualize critical text keywords influencing job classification to explain *why* a posting was flagged.
        * **Multi-model Benchmarking**: Support cross-validated comparisons across Logistic Regression, Naive Bayes, Linear SVM, Decision Trees, Random Forests, and XGBoost.
        """)
        
        st.subheader("💡 Key Risk Indicators in Fake Listings")
        st.markdown("""
        1. **Vague Requirements & Profiles**: Scammers often write generic job descriptions to attract a wider pool of unsuspecting victims.
        2. **Financial Requests**: Legitimate employers will never ask for payment, training fees, or background check purchases up front.
        3. **Missing Company Logo or Website**: A significant percentage of fraudulent jobs are associated with incomplete profiles or lack company logos.
        4. **Irregular Email Domains**: Contact information utilizing free webmail (like Gmail or Yahoo) instead of institutional domains.
        """)
        
    with col_right:
        st.subheader("📊 Dataset Overview")
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            st.dataframe(df[['title', 'location', 'company_profile', 'fraudulent']].head(8), height=280)
        else:
            st.info("Dataset is not loaded locally yet. Run training to download it.")

# 6. PAGE 2: EDA Gallery
elif navigation == "EDA Gallery":
    st.markdown("""
    <div class='header-box'>
        <h1 class='header-title'>Exploratory Data Analysis</h1>
        <p class='header-subtitle'>Statistical analysis and visual breakdowns of features in the fake job postings corpus.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not os.path.exists(os.path.join(ASSETS_DIR, "class_distribution.png")):
        st.warning("⚠️ EDA plots are not generated yet. Please navigate to the 'Model Comparison' tab and trigger model training to generate the plots.")
    else:
        tab1, tab2, tab3 = st.tabs(["📉 Distributions & Correlations", "🔠 WordClouds & Text Frequencies", "🗺️ Hiring Demographics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Class Imbalance Distribution")
                st.image(os.path.join(ASSETS_DIR, "class_distribution.png"), use_container_width=True)
                st.caption("The dataset contains roughly ~17,000 legitimate jobs vs ~860 fake jobs (high class imbalance).")
            with col2:
                st.write("##### Metadata Correlation Heatmap")
                if os.path.exists(os.path.join(ASSETS_DIR, "correlation_heatmap.png")):
                    st.image(os.path.join(ASSETS_DIR, "correlation_heatmap.png"), use_container_width=True)
                    st.caption("Correlation of features like 'has_company_logo', 'has_questions', and 'telecommuting' with fraudulent label.")
                else:
                    st.info("Correlation plot skipped.")
                    
            st.write("##### Missing Values Heatmap")
            st.image(os.path.join(ASSETS_DIR, "missing_value_heatmap.png"), use_container_width=True)
            st.caption("Visualizing missing values in raw features (yellow lines show null elements).")

        with tab2:
            st.write("##### Top 20 Most Frequent Cleaned Words")
            st.image(os.path.join(ASSETS_DIR, "word_frequency.png"), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Legitimate Job WordCloud")
                st.image(os.path.join(ASSETS_DIR, "wordcloud_legitimate.png"), use_container_width=True)
            with col2:
                st.write("##### Fraudulent Job WordCloud")
                st.image(os.path.join(ASSETS_DIR, "wordcloud_fraudulent.png"), use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Top 10 Hiring Locations")
                st.image(os.path.join(ASSETS_DIR, "top_locations.png"), use_container_width=True)
            with col2:
                st.write("##### Employment Type Distribution")
                st.image(os.path.join(ASSETS_DIR, "employment_types.png"), use_container_width=True)

# 7. PAGE 3: Model Comparison & Training
elif navigation == "Model Comparison & Training":
    st.markdown("""
    <div class='header-box'>
        <h1 class='header-title'>Model Training & Parameter Comparison</h1>
        <p class='header-subtitle'>Benchmarking multiple classifiers utilizing GridSearchCV. Tweak parameters and trigger new training cycles directly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_control, col_results = st.columns([1, 2])
    
    with col_control:
        st.subheader("⚙️ Training Configurations")
        v_type = st.selectbox("Feature Engineering Vectorizer", ["tfidf", "count"], index=0, format_func=lambda x: "TF-IDF Vectorizer" if x == "tfidf" else "Bag of Words (CountVectorizer)")
        sample_check = st.checkbox("Fast Sample Mode (5,000 records)", value=False, help="Recommended. Restricts records to speed up GridSearchCV training locally.")
        
        train_btn = st.button("🚀 Train All Models & Tune")
        
        if train_btn:
            with st.spinner("Executing ML pipeline (Download -> Preprocess -> Grid Search -> Evaluate)... This might take a few minutes."):
                try:
                    best_name, comparison_df = run_training_pipeline(vectorizer_type=v_type, sample_data=sample_check)
                    st.success(f"🎉 Training complete! Best model identified: **{best_name}**")
                    # Force page reload of cache
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during training pipeline: {e}")
                    
        st.markdown("---")
        st.info("💡 **Training Note**: Stratified 3-fold cross validation is applied. The models optimize hyperparameter criteria like regularization bounds ('C') and Laplace smoothing bounds ('alpha') based on the F1-Score metric.")
        
    with col_results:
        st.subheader("📊 Performance Comparison Table")
        comparison_path = os.path.join(MODELS_DIR, "model_comparison.json")
        if os.path.exists(comparison_path):
            with open(comparison_path, 'r') as f:
                comp_data = json.load(f)
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            
            best_idx = comp_df['F1-Score'].idxmax()
            best_m = comp_df.loc[best_idx, 'Model Name']
            best_f1 = comp_df.loc[best_idx, 'F1-Score']
            
            st.success(f"🏆 **Best Performing Model**: `{best_m}` (F1-Score: **{best_f1}**)")
            
            # Show evaluation curves of best model if available
            st.markdown("### Best Model Visual Metrics")
            model_slug = best_m.replace(" ", "_").lower()
            
            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                cm_plot_path = os.path.join(ASSETS_DIR, f"cm_{model_slug}.png")
                if os.path.exists(cm_plot_path):
                    st.image(cm_plot_path, caption="Confusion Matrix Heatmap")
            with col_plot2:
                roc_plot_path = os.path.join(ASSETS_DIR, f"roc_{model_slug}.png")
                if os.path.exists(roc_plot_path):
                    st.image(roc_plot_path, caption="Receiver Operating Characteristic (ROC) Curve")
        else:
            st.info("No trained models logs found. Run the training pipeline to generate metrics.")

# 8. PAGE 4: Real-time Predictor
elif navigation == "Real-time Predictor":
    st.markdown("""
    <div class='header-box'>
        <h1 class='header-title'>Real-time Job Posting Verification</h1>
        <p class='header-subtitle'>Input a job posting's text fields below to evaluate the likelihood of it being fraudulent.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if predictor is None:
        st.error("❌ Prediction engine not loaded. Please run model training on the 'Model Comparison' page first.")
    else:
        # Load mock template listings for quick testing
        st.markdown("### 📝 Quick Test Templates")
        mock_selection = st.selectbox(
            "Load a Mock job posting to pre-fill inputs:",
            ["Blank Form", "Legitimate Software Engineer (Tech Corp)", "Scam Administrative Assistant (DataEntry Inc)", "Scam Mystery Shopper Offer"]
        )
        
        # Define mock content
        mock_data = {
            "Blank Form": {"title": "", "location": "", "company_profile": "", "description": "", "requirements": "", "benefits": ""},
            "Legitimate Software Engineer (Tech Corp)": {
                "title": "Senior Python Backend Developer",
                "location": "US, CA, San Francisco",
                "company_profile": "Tech Corp is an established market leader in cloud computing and data analytics. Established in 2015, we support over 10 million daily active users across global enterprises.",
                "description": "We are seeking a senior Python developer to join our backend engineering team. You will build and scale REST APIs, design relational and NoSQL databases, and coordinate with frontend developers. Working environment includes modern CI/CD pipelines.",
                "requirements": "Minimum 5 years of professional backend software development experience. Proficient in Python, Django/FastAPI, PostgreSQL, Docker, and AWS. BS degree in Computer Science or equivalent field.",
                "benefits": "Competitive market salary. Comprehensive medical, dental, and vision insurance. 401(k) retirement matching. Standard paid time off."
            },
            "Scam Administrative Assistant (DataEntry Inc)": {
                "title": "Virtual Data Entry / Administrative Assistant",
                "location": "US, NY, New York",
                "company_profile": "A young start-up focusing on data solutions globally.",
                "description": "URGENTLY HIRING! Work from home part-time. We are seeking a reliable remote assistant to process data entry sheets and respond to customer requests. Immediate start! Payments are sent weekly via bank transfers. Training provided.",
                "requirements": "Must have access to a computer and high-speed internet. Able to work independently with minimal supervision. No prior experience required, we accept entry-level candidates.",
                "benefits": "High hourly pay ($35-$45/hour). Flexible working hours. Easy tasks. Weekly payouts."
            },
            "Scam Mystery Shopper Offer": {
                "title": "Mystery Shopper / Evaluation Coordinator",
                "location": "US, FL, Miami",
                "company_profile": "A marketing research agency looking for evaluation representatives.",
                "description": "Earn money while shopping! We need shoppers to evaluate local retail stores, restaurant chains, and customer services. You will receive assignments and check cards to purchase items, evaluate, and provide reports.",
                "requirements": "Excellent written communication skills. Access to transport. Ability to follow instructions precisely.",
                "benefits": "Commission fee per assignment ($150-$200). Funding check sent in advance for purchases."
            }
        }
        
        selected_mock = mock_data[mock_selection]
        
        # User input fields
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Job Title", value=selected_mock["title"], placeholder="e.g. Senior Software Engineer")
                location = st.text_input("Location", value=selected_mock["location"], placeholder="e.g. US, NY, New York")
                company_profile = st.text_area("Company Profile", value=selected_mock["company_profile"], placeholder="Provide a brief summary of the hiring organization...")
            with col2:
                description = st.text_area("Job Description", value=selected_mock["description"], placeholder="Detailed description of the responsibilities...")
                requirements = st.text_area("Requirements", value=selected_mock["requirements"], placeholder="Qualifications, technical skills, and experience needed...")
                benefits = st.text_area("Benefits", value=selected_mock["benefits"], placeholder="Compensation package, health insurance, PTO, etc...")
                
            predict_btn = st.form_submit_button("🔍 Verify Job Posting")
            
        if predict_btn:
            # Input Validation
            if not title.strip() or not description.strip():
                st.warning("⚠️ Both **Job Title** and **Job Description** are required to perform analysis.")
            else:
                with st.spinner("Analyzing text markers using NLP engine..."):
                    job_data = {
                        "title": title,
                        "location": location,
                        "company_profile": company_profile,
                        "description": description,
                        "requirements": requirements,
                        "benefits": benefits
                    }
                    
                    prediction, confidence, keywords = predictor.predict(job_data)
                    
                    # Save prediction to history log
                    save_prediction_to_history(job_data, prediction, confidence)
                    
                    # Display prediction results
                    st.markdown("### 📈 Analysis Verdict")
                    
                    if prediction == "Fraudulent":
                        st.markdown(f"""
                        <div class='alert-card alert-fraud'>
                            <h2 style='margin:0; color:#ef4444;'>❌ Fraudulent Job Posting Detected</h2>
                            <p style='margin:10px 0 0 0; font-size:15px; color:#cbd5e1;'>Confidence Score: <b>{confidence:.2%}</b>. High risk indicators are present in the text structure.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='alert-card alert-legit'>
                            <h2 style='margin:0; color:#10b981;'>🟢 Legitimate Job Posting</h2>
                            <p style='margin:10px 0 0 0; font-size:15px; color:#cbd5e1;'>Confidence Score: <b>{confidence:.2%}</b>. Text aligns with standard authentic listings.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Keywords Chart
                    if keywords:
                        st.markdown("### 🏷️ Critical Keyword Influences")
                        st.write("Keywords that had the most influence on the model decision (Positive weights push towards FRAUD, Negative weights push towards LEGITIMATE):")
                        
                        kw_df = pd.DataFrame(keywords, columns=['Keyword', 'Influence Weight'])
                        kw_df['Sign'] = kw_df['Influence Weight'].apply(lambda x: 'Red' if x > 0 else 'Blue')
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = ['#ef4444' if x > 0 else '#3b82f6' for x in kw_df['Influence Weight']]
                        ax.barh(kw_df['Keyword'], kw_df['Influence Weight'], color=colors)
                        ax.invert_yaxis()
                        ax.set_xlabel("Feature Coefficient Weight", color='#cbd5e1')
                        ax.tick_params(colors='#cbd5e1')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    # Report Exports
                    pdf_filename = f"verif_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_path = os.path.join(MODELS_DIR, pdf_filename)
                    generate_pdf_report(job_data, prediction, confidence, keywords, pdf_path)
                    
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                        
                    st.download_button(
                        label="📄 Download Prediction PDF Report",
                        data=pdf_data,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )

# 9. PAGE 5: History Log
elif navigation == "History Log":
    st.markdown("""
    <div class='header-box'>
        <h1 class='header-title'>Verification History Logs</h1>
        <p class='header-subtitle'>View, inspect, and export previous analysis results generated during your session.</p>
    </div>
    """, unsafe_allow_html=True)
    
    history_df = get_prediction_history()
    
    if history_df.empty:
        st.info("No queries analyzed yet. Navigate to the 'Real-time Predictor' page and verify a job posting to populate this log.")
    else:
        st.subheader("📋 Session History List")
        st.dataframe(history_df, use_container_width=True)
        
        # Download csv history
        with open(HISTORY_PATH, "r") as f:
            csv_data = f.read()
            
        st.download_button(
            label="📥 Export History as CSV",
            data=csv_data,
            file_name="fake_job_prediction_history.csv",
            mime="text/csv"
        )

# 10. PAGE 6: About & Architecture
elif navigation == "About & Architecture":
    st.markdown("""
    <div class='header-box'>
        <h1 class='header-title'>Architecture & Methodology</h1>
        <p class='header-subtitle'>A complete breakdown of the data processing and training pipelines governing this system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🏗️ System Architecture Flow")
    st.markdown("""
    ```
    +--------------------------------------------------------+
    |                    Streamlit UI                        |
    |      (Input forms, history log, reports, Retraining)   |
    +-----------------------------------------------+--------+
                                                    |
                                             Predicts on input
                                                    |
                                                    v
    +--------------------------------------------------------+
    |                 NLP Preprocessing                      |
    |  (Text Merge -> Lowercase -> Cleansing -> Lemmatize)   |
    +-----------------------------------------------+--------+
                                                    |
                                            Vectorized features
                                                    |
                                                    v
    +--------------------------------------------------------+
    |            Vectorizer & Model Persistence              |
    |     (loads joblib vectorizers & tuned classifier)      |
    +-----------------------------------------------+--------+
                                                    |
                                            Predictions & proba
                                                    |
                                                    v
    +--------------------------------------------------------+
    |                    PDF Generator                       |
    |      (Outputs styled visual verification report)        |
    +--------------------------------------------------------+
    ```
    """)
    
    st.subheader("📚 Dataset Details")
    st.markdown("""
    * **Origin**: Employment Scam Aegean Dataset (EMSCAD).
    * **Distribution**: ~18k employment postings collected from 2012 to 2014.
    * **Target Label**: `fraudulent` (1 = scam listing, 0 = genuine job).
    * **Text Fields**: Includes job postings' titles, locations, descriptions, requirements, benefits, and company profiles.
    """)
    
    st.subheader("🔧 Preprocessing & NLP Pipeline")
    st.markdown("""
    * **Feature Fusion**: Text values across six different fields are aggregated to form a unified textual corpus per listing.
    * **RegEx Cleansing**: Removes non-alphabetic characters, digits, email addresses, and hyperlink URLs.
    * **Tokenization & Stopwords Filtering**: Filters out common English words and boilerplate text.
    * **Lemmatization**: Reduces inflected words to their root forms (e.g. "employees" -> "employee", "running" -> "run") using NLTK WordNet.
    * **Feature Vectorization**: Compares TF-IDF weighting (importance scoring) with CountVectorizer (bag of words bag counts).
    """)
