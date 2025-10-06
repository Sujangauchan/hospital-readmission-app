import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import joblib
import re
from datasets import Dataset
import shap
import matplotlib.pyplot as plt
import io
from PIL import Image

# page config
st.set_page_config(
    page_title="Readmission Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# custom css
st.markdown("""
<style>
    /* color palette */
    :root {
        --light-blue: #B5C7D2;
        --dark-blue: #88A3B1;
        --darkest-blue: #467C9B;
    }
    
    /* main container */
    .main {
        background-color: #FFFFFF;
    }
    
    /* header styling */
    .header-container {
        background: linear-gradient(135deg, #467C9B 0%, #88A3B1 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* section headers */
    .section-header {
        background-color: rgba(181, 199, 210, 0.15);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #467C9B;
        margin: 1.5rem 0 1rem 0;
    }
    
    .section-header h3 {
        color: #467C9B;
        margin: 0;
        font-size: 1.3rem;
    }
    
    /* predict button */
    .stButton > button {
        background-color: #467C9B;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 3rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #88A3B1;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(70, 124, 155, 0.3);
    }
    
    /* risk badge */
    .risk-badge {
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .risk-high {
        background-color: #dc3545;
        color: white;
    }
    
    .risk-medium {
        background-color: #ffc107;
        color: #000;
    }
    
    .risk-low {
        background-color: #28a745;
        color: white;
    }
    
    /* result cards */
    .result-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid;
        margin: 1rem 0;
    }
    
    .result-card.meta {
        border-color: #467C9B;
        background: linear-gradient(135deg, #467C9B10 0%, #88A3B120 100%);
    }
    
    .result-card.xgb {
        border-color: #88A3B1;
    }
    
    .result-card.bert {
        border-color: #B5C7D2;
    }
    
    .result-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .result-meta .result-title {
        font-size: 1.8rem;
        color: #467C9B;
    }
    
    .result-xgb .result-title {
        font-size: 1.5rem;
        color: #88A3B1;
    }
    
    .result-bert .result-title {
        font-size: 1.3rem;
        color: #467C9B;
    }
    
    .result-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .result-meta .result-value {
        font-size: 3.5rem;
    }
    
    .result-xgb .result-value {
        font-size: 2.8rem;
    }
    
    /* footer */
    .footer {
        background-color: rgba(181, 199, 210, 0.1);
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 3rem;
        border-left: 4px solid #ffc107;
    }
    
    /* input styling */
    .stSelectbox, .stNumberInput, .stTextArea {
        margin-bottom: 0.5rem;
    }
    
    /* expander */
    .streamlit-expanderHeader {
        background-color: rgba(181, 199, 210, 0.1);
        border-radius: 5px;
        font-weight: 600;
        color: #467C9B;
    }
</style>
""", unsafe_allow_html=True)

# load models
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bert_tokenizer = AutoTokenizer.from_pretrained("sujangauchan/DISCHARGY_SUMMARY_BERT_READMISSION_CLASSIFIER")
    bert_model = AutoModelForSequenceClassification.from_pretrained("sujangauchan/DISCHARGY_SUMMARY_BERT_READMISSION_CLASSIFIER")
    bert_model.to(device)
    bert_model.eval()
    
    xgb_model = joblib.load('models/best_readmission_model_xgboost.pkl')
    meta_model = joblib.load('models/meta_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    background_data = pd.read_csv('models/background_data.csv')
    
    return bert_tokenizer, bert_model, xgb_model, meta_model, preprocessor, background_data, device

# Helper function to get feature names from old preprocessor
def get_feature_names_from_preprocessor(preprocessor, input_df):
    """
    Extract feature names from a preprocessor, handling both old and new scikit-learn versions
    """
    try:
        # Try the new method first
        return preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older scikit-learn versions
        feature_names = []
        
        # Get the transformers
        transformers = preprocessor.transformers_
        
        for name, transformer, columns in transformers:
            if name == 'num':
                # Numeric features - use column names directly
                feature_names.extend(columns)
            elif name == 'cat':
                # Categorical features - get encoded names
                try:
                    # Try to get the OneHotEncoder from the pipeline
                    if hasattr(transformer, 'named_steps'):
                        encoder = transformer.named_steps.get('encoder', transformer.named_steps.get('onehot', None))
                    else:
                        encoder = transformer
                    
                    if encoder is not None and hasattr(encoder, 'categories_'):
                        for i, col in enumerate(columns):
                            if i < len(encoder.categories_):
                                for cat in encoder.categories_[i]:
                                    feature_names.append(f"{col}_{cat}")
                    else:
                        # If we can't get categories, just use column names
                        feature_names.extend(columns)
                except Exception as e:
                    st.warning(f"Could not extract categorical feature names: {e}")
                    feature_names.extend(columns)
        
        return feature_names

# text preprocessing
def preprocess1(x):
    y = re.sub(r'\[(.*?)\]', '', x)
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'dr\.', 'doctor', y)
    y = re.sub(r'm\.d\.', 'md', y)
    y = re.sub(r'--|__|==|_', '', y)
    y = re.sub(r'name:', '', y)
    y = re.sub(r'unit no:', '', y)
    y = re.sub(r'admission date:', '', y)
    y = re.sub(r'discharge date:', '', y)
    y = re.sub(r'date of birth:', '', y)
    y = re.sub(r'attending: .*?\n', '', y)
    return y

def preprocessing_bert(text, procedures_codes, diagnoses_codes):
    combined_text = f"{procedures_codes} {diagnoses_codes} {text}"
    combined_text = combined_text.replace('\n', ' ').replace('\r', ' ').strip().lower()
    combined_text = preprocess1(combined_text)
    
    words = combined_text.split()
    chunks = []
    n = len(words) // 128
    
    for j in range(n):
        chunks.append(' '.join(words[j*128:(j+1)*128]))
    
    leftover = len(words) % 128
    if leftover > 10:
        chunks.append(' '.join(words[-leftover:]))
    
    return chunks

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

def predict_bert(chunks, model, tokenizer, device, batch_size=32):
    if not chunks:
        return 0.5
    
    chunk_df = pd.DataFrame({'text': chunks})
    chunk_dataset = Dataset.from_pandas(chunk_df)
    tokenized_chunks = chunk_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    all_probs = []
    
    for i in range(0, len(tokenized_chunks), batch_size):
        batch_data = tokenized_chunks[i:i+batch_size]
        batch_examples = []
        
        for j in range(len(batch_data['input_ids'])):
            example = {
                'input_ids': batch_data['input_ids'][j],
                'attention_mask': batch_data['attention_mask'][j]
            }
            batch_examples.append(example)
        
        batch = data_collator(batch_examples)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            positive_probs = probabilities[:, 1].cpu().numpy()
            all_probs.extend(positive_probs)
    
    return np.mean(all_probs)

def predict_xgboost(input_df, preprocessor, model):
    X_transformed = preprocessor.transform(input_df)
    y_pred_proba = model.predict_proba(X_transformed)[:, 1]
    return y_pred_proba[0], X_transformed

def get_risk_label(probability):
    if probability >= 0.6:
        return "HIGH RISK", "risk-high"
    elif probability >= 0.4:
        return "MEDIUM RISK", "risk-medium"
    else:
        return "LOW RISK", "risk-low"

# header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🏥 READMISSION RISK ASSESSMENT SYSTEM</h1>
    <p class="header-subtitle">Advanced AI-Powered 30-Day Readmission Prediction</p>
</div>
""", unsafe_allow_html=True)

# categorical dropdowns
ADMISSION_TYPES = ["Select...", "EW EMER.", "OBSERVATION ADMIT", "EU OBSERVATION", "SURGICAL SAME DAY ADMISSION", 
                   "URGENT", "DIRECT EMER.", "DIRECT OBSERVATION", "ELECTIVE", "AMBULATORY OBSERVATION"]

ADMISSION_LOCATIONS = ["Select...", "EMERGENCY ROOM", "PHYSICIAN REFERRAL", "TRANSFER FROM HOSPITAL", 
                       "WALK-IN/SELF REFERRAL", "CLINIC REFERRAL", "PROCEDURE SITE", "PACU", 
                       "INTERNAL TRANSFER TO OR FROM PSYCH", "TRANSFER FROM SKILLED NURSING FACILITY", 
                       "INFORMATION NOT AVAILABLE", "AMBULATORY SURGERY TRANSFER"]

INSURANCE_TYPES = ["Select...", "Medicare", "Private", "Medicaid", "Other", "No charge"]

GENDERS = ["Select...", "F", "M"]

LANGUAGES = ["Select...", "English", "Other", "Spanish", "Russian", "Chinese"]

MARITAL_STATUSES = ["Select...", "MARRIED", "SINGLE", "WIDOWED", "DIVORCED", "Unknown"]

RACES = ["Select...", "WHITE", "Other", "BLACK/AFRICAN AMERICAN", "OTHER", "UNKNOWN"]

# patient information
with st.expander("📋 PATIENT INFORMATION", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", GENDERS)
        race = st.selectbox("Race", RACES)
        language = st.selectbox("Language", LANGUAGES)
        marital_status = st.selectbox("Marital Status", MARITAL_STATUSES)
    
    with col2:
        st.markdown("**Administrative**")
        admission_type = st.selectbox("Admission Type", ADMISSION_TYPES)
        admission_location = st.selectbox("Admission Location", ADMISSION_LOCATIONS)
        insurance = st.selectbox("Insurance", INSURANCE_TYPES)

# clinical information
with st.expander("🏥 CLINICAL INFORMATION", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Laboratory Data**")
        total_lab_events = st.number_input("Total Lab Events", min_value=0, value=0, step=1)
        normal_counts = st.number_input("Normal Lab Results", min_value=0, value=0, step=1)
        abnormal_counts = st.number_input("Abnormal Lab Results", min_value=0, value=0, step=1)
        abnormal_percent = st.number_input("Abnormal % (0-100)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        lab_abnormal_ratio = abnormal_counts / (total_lab_events + 1) if total_lab_events > 0 else 0
        high_lab_volume = 1 if total_lab_events > 100 else 0
    
    with col2:
        st.markdown("**Procedures & Diagnoses**")
        procedures_total = st.number_input("Total Procedures", min_value=0, value=0, step=1)
        diagnoses_total = st.number_input("Total Diagnoses", min_value=0, value=0, step=1)
        had_procedures = 1 if procedures_total > 0 else 0
        had_labs = 1 if total_lab_events > 0 else 0
        had_diagnoses = 1 if diagnoses_total > 0 else 0
        high_procedure_count = 1 if procedures_total > 5 else 0
        high_diagnosis_count = 1 if diagnoses_total > 10 else 0
    
    with col3:
        st.markdown("**Temporal Features**")
        admission_hour = st.number_input("Admission Hour (0-23)", min_value=0, max_value=23, value=0, step=1)
        admission_day = st.number_input("Admission Day of Week (0-6)", min_value=0, max_value=6, value=0, step=1)
        weekend_admission = 1 if admission_day >= 5 else 0
        night_admission = 1 if (admission_hour >= 18 or admission_hour <= 6) else 0

# stay and history
with st.expander("📊 STAY & HISTORY DETAILS", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Stay Duration**")
        stay_duration = st.number_input("Stay Duration (days)", min_value=0.0, value=0.0, step=0.1)
        ed_duration = st.number_input("ED Duration (days)", min_value=0.0, value=0.0, step=0.1)
        visited_ed = 1 if ed_duration > 0 else 0
        
        short_stay = 1 if stay_duration < 1 else 0
        medium_stay = 1 if 1 <= stay_duration <= 7 else 0
        long_stay = 1 if stay_duration > 7 else 0
        very_long_stay = 1 if stay_duration > 14 else 0
    
    with col2:
        st.markdown("**Admission History**")
        previous_admissions = st.number_input("Previous Admissions", min_value=0, value=0, step=1)
        days_since_last = st.number_input("Days Since Last Admission", min_value=0, value=0, step=1)
        total_admissions_ever = st.number_input("Total Admissions Ever", min_value=0, value=0, step=1)
        
        admitted_before = 1 if previous_admissions > 0 else 0
        frequent_patient = 1 if total_admissions_ever >= 3 else 0

# calculated per day features
if stay_duration > 0:
    procedures_per_day = procedures_total / (stay_duration + 0.1)
    labs_per_day = total_lab_events / (stay_duration + 0.1)
    diagnoses_per_day = diagnoses_total / (stay_duration + 0.1)
else:
    procedures_per_day = 0
    labs_per_day = 0
    diagnoses_per_day = 0

emergency_admission = 1 if admission_type in ['EW EMER.', 'URGENT', 'DIRECT EMER.'] else 0

# clinical notes
with st.expander("📝 CLINICAL NOTES & CODES", expanded=True):
    clinical_text = st.text_area("Clinical Notes (Discharge Summary)", height=200, 
                                  placeholder="Enter discharge summary and clinical notes here...")
    
    col1, col2 = st.columns(2)
    with col1:
        procedures_codes = st.text_input("Procedure ICD Codes", placeholder="e.g., 9904 3995 8856")
    with col2:
        diagnoses_codes = st.text_input("Diagnosis ICD Codes", placeholder="e.g., 41401 4280 25000")

# predict button
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("🔬 PREDICT READMISSION RISK", use_container_width=True)

if predict_button:
    # validation
    if any(x == "Select..." for x in [gender, race, language, marital_status, admission_type, admission_location, insurance]):
        st.error("Please select values for all dropdown fields.")
    elif not clinical_text.strip():
        st.error("Please enter clinical notes.")
    else:
        with st.spinner("Analyzing patient data..."):
            # load models
            bert_tokenizer, bert_model, xgb_model, meta_model, preprocessor, background_data, device = load_models()
            
            # prepare tabular data
            input_data = {
                'admission_type': admission_type,
                'admission_location': admission_location,
                'insurance': insurance,
                'gender': gender,
                'language_grouped': language,
                'marital_status': marital_status,
                'race_grouped': race,
                'diagnoses_total_count': diagnoses_total,
                'procedures_total_count': procedures_total,
                'Normal_counts': normal_counts,
                'Abnormal_counts': abnormal_counts,
                'Total_lab_events': total_lab_events,
                'abnormal%': abnormal_percent,
                'stay_duration_days': stay_duration,
                'ed_duration_days': ed_duration,
                'previous_admissions': previous_admissions,
                'days_since_last_admission': days_since_last,
                'admitted_before': admitted_before,
                'visited_ed': visited_ed,
                'had_procedures': had_procedures,
                'had_labs': had_labs,
                'had_diagnoses': had_diagnoses,
                'admission_hour': admission_hour,
                'admission_day_of_week': admission_day,
                'weekend_admission': weekend_admission,
                'night_admission': night_admission,
                'total_admissions_ever': total_admissions_ever,
                'frequent_patient': frequent_patient,
                'high_lab_volume': high_lab_volume,
                'high_procedure_count': high_procedure_count,
                'high_diagnosis_count': high_diagnosis_count,
                'short_stay': short_stay,
                'medium_stay': medium_stay,
                'long_stay': long_stay,
                'procedures_per_day': procedures_per_day,
                'labs_per_day': labs_per_day,
                'diagnoses_per_day': diagnoses_per_day,
                'abnormal_lab_burden': lab_abnormal_ratio,
                'very_long_stay': very_long_stay,
                'emergency_admission': emergency_admission
            }
            
            input_df = pd.DataFrame([input_data])
            
            # xgboost prediction
            st.info("Processing tabular features with XGBoost...")
            xgb_proba, X_transformed = predict_xgboost(input_df, preprocessor, xgb_model)
            xgb_pred = 1 if xgb_proba >= 0.5 else 0
            
            # bert prediction
            st.info("Analyzing clinical notes with BERT...")
            chunks = preprocessing_bert(clinical_text, procedures_codes, diagnoses_codes)
            bert_proba = predict_bert(chunks, bert_model, bert_tokenizer, device)
            bert_pred = 1 if bert_proba >= 0.5 else 0
            
            # meta prediction
            st.info("Computing ensemble prediction...")
            X_meta = np.array([[xgb_proba, xgb_pred, bert_proba, bert_pred]])
            meta_pred = meta_model.predict(X_meta)[0]
            meta_proba = meta_model.predict_proba(X_meta)[0, 1]
            
            # results section
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><h3>🎯 RISK ASSESSMENT RESULTS</h3></div>', unsafe_allow_html=True)
            
            # risk badge
            risk_label, risk_class = get_risk_label(meta_proba)
            st.markdown(f'<div class="risk-badge {risk_class}">{risk_label}</div>', unsafe_allow_html=True)
            
            # ensemble result
            st.markdown(f"""
            <div class="result-card result-meta meta">
                <div class="result-title">ENSEMBLE MODEL PREDICTION (META LEARNER)</div>
                <div class="result-value" style="color: #467C9B;">{meta_proba*100:.1f}%</div>
                <div style="font-size: 1.2rem; margin-top: 0.5rem;">
                    Prediction: <strong>{"READMISSION" if meta_pred == 1 else "NO READMISSION"}</strong>
                </div>
                <div style="margin-top: 1rem; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden;">
                    <div style="height: 100%; width: {meta_proba*100}%; background: linear-gradient(90deg, #467C9B, #88A3B1);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # xgboost result
                st.markdown(f"""
                <div class="result-card result-xgb xgb">
                    <div class="result-title">XGBoost Model (Tabular Data)</div>
                    <div class="result-value" style="color: #88A3B1;">{xgb_proba*100:.1f}%</div>
                    <div style="font-size: 1.1rem; margin-top: 0.5rem;">
                        Prediction: <strong>{"READMISSION" if xgb_pred == 1 else "NO READMISSION"}</strong>
                    </div>
                    <div style="margin-top: 1rem; height: 15px; background: #e9ecef; border-radius: 8px; overflow: hidden;">
                        <div style="height: 100%; width: {xgb_proba*100}%; background: #88A3B1;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # bert result
                st.markdown(f"""
                <div class="result-card result-bert bert">
                    <div class="result-title">BERT Model (Clinical Text)</div>
                    <div class="result-value" style="color: #467C9B;">{bert_proba*100:.1f}%</div>
                    <div style="font-size: 1.1rem; margin-top: 0.5rem;">
                        Prediction: <strong>{"READMISSION" if bert_pred == 1 else "NO READMISSION"}</strong>
                    </div>
                    <div style="margin-top: 1rem; height: 15px; background: #e9ecef; border-radius: 8px; overflow: hidden;">
                        <div style="height: 100%; width: {bert_proba*100}%; background: #B5C7D2;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # shap waterfall
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><h3>📈 FEATURE IMPACT ANALYSIS (SHAP)</h3></div>', unsafe_allow_html=True)
            st.markdown("**Understanding Key Risk Factors (XGBoost Model)**")
            
            with st.spinner("Generating SHAP analysis..."):
                X_background = preprocessor.transform(background_data)
                explainer = shap.TreeExplainer(xgb_model, X_background)
                shap_values = explainer.shap_values(X_transformed)
                
                # Get feature names with fallback for version compatibility
                feature_names = get_feature_names_from_preprocessor(preprocessor, input_df)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=X_transformed[0],
                        feature_names=feature_names
                    ),
                    max_display=15,
                    show=False
                )
                st.pyplot(fig)
                plt.close()
            
            st.markdown("""
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(181, 199, 210, 0.1); border-radius: 8px;">
                <p style="margin: 0; color: #467C9B; font-size: 0.95rem;">
                    <strong>Interpretation:</strong> The waterfall chart shows how each feature contributed to the final prediction. 
                    Features pushing right (red) increase readmission risk, while features pushing left (blue) decrease risk.
                    The chart displays the top 15 most impactful features for this patient.
                </p>
            </div>
            """, unsafe_allow_html=True)

# footer
st.markdown("""
<div class="footer">
    <strong>⚠️ Clinical Decision Support Tool</strong><br>
    This system is intended for use by qualified healthcare professionals only. 
    Predictions should be used alongside clinical judgment, not as sole determinant of care.
    The AI models provide risk assessment based on historical data patterns and should complement, not replace, professional medical expertise.
</div>
""", unsafe_allow_html=True)