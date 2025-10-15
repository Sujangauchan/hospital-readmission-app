# Hospital Readmission Risk Assessment System

A professional Streamlit web application for predicting 30-day hospital readmission risk using ensemble learning (XGBoost + BERT + Logistic Regression meta-learner) with SHAP interpretability.

---

# Features

- **Multi-Model Prediction**: Ensemble of XGBoost (tabular data) + BERT (clinical text) + Meta-learner
- **Risk Stratification**: Automatic classification into Low/Medium/High risk categories
- **SHAP Interpretability**: Waterfall charts showing feature impact on predictions
- **Professional UI**: Clean, minimal design suitable for clinical settings
- **Real-time Prediction**: Instant risk assessment after data entry

---

# Local Setup Instructions

### FILE STRUCTURE
```
hospital-readmission-app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── models/                         # Model files directory
    ├── best_readmission_model_xgboost.pkl
    ├── meta_model.pkl
    ├── preprocessor.pkl
    └── background_data.csv
```
---

## Using the Application

### Input Data Entry

1. **Patient Information**
   - Select demographic details (gender, race, language, marital status)
   - Select administrative details (admission type, location, insurance)

2. **Clinical Information**
   - Enter laboratory data (total labs, normal/abnormal counts)
   - Enter procedure and diagnosis counts
   - Temporal features are auto-calculated

3. **Stay & History Details**
   - Enter stay duration and ED duration
   - Provide admission history (previous admissions, frequency)

4. **Clinical Notes & Codes**
   - Paste discharge summary in the text area
   - Enter procedure and diagnosis ICD codes (space-separated)

5. **Click "PREDICT READMISSION RISK"**

### Understanding Results

**Risk Categories:**
- 🟢 **LOW RISK**: < 40% probability
- 🟡 **MEDIUM RISK**: 40-60% probability
- 🔴 **HIGH RISK**: > 60% probability

**Model Predictions:**
1. **Ensemble Model (Meta)**: Combined prediction (LARGEST display)
2. **XGBoost Model**: Tabular data prediction (SECOND)
3. **BERT Model**: Clinical text prediction (THIRD)

**SHAP Waterfall Chart:**
- Shows top 15 features impacting the prediction
- Red bars: Increase readmission risk
- Blue bars: Decrease readmission risk
- Magnitude shows strength of impact

---

##  Model Information

### XGBoost Model
- **Input**: 42 tabular features (demographics, clinical, temporal)
- **Output**: Readmission probability
- **File size**: 2.65 MB

### BERT Model
- **Type**: BioBERT fine-tuned for readmission prediction
- **Input**: Clinical text (discharge summaries + ICD codes)
- **Output**: Readmission probability
- **Source**: HuggingFace Hub
- **Model ID**: `sujangauchan/DISCHARGY_SUMMARY_BERT_READMISSION_CLASSIFIER`

### Meta Model (Logistic Regression)
- **Input**: Predictions from XGBoost + BERT (4 features)
- **Output**: Final ensemble readmission probability
- **File size**: 911 bytes

### Preprocessor
- **Contains**: OneHotEncoder + StandardScaler for tabular features
- **File size**: ~50 KB

### Background Data
- **Purpose**: SHAP baseline for XGBoost interpretability
- **Size**: 100 patient records
- **Format**: CSV
