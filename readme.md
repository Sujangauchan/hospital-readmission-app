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

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Step 1: Create Project Directory

```bash
# Navigate to where you want the project
cd Desktop  # or your preferred location

# Create project folder
mkdir hospital-readmission-app
cd hospital-readmission-app
```

### Step 2: Set Up File Structure

Create the following structure:

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

**Create the models folder:**

```bash
mkdir models
```

**Move your 4 files from Desktop/Files to the models folder:**

```bash
# On Windows (Command Prompt)
move "C:\Users\YourUsername\Desktop\Files\best_readmission_model_xgboost.pkl" models\
move "C:\Users\YourUsername\Desktop\Files\meta_model.pkl" models\
move "C:\Users\YourUsername\Desktop\Files\preprocessor.pkl" models\
move "C:\Users\YourUsername\Desktop\Files\background_data.csv" models\

# On Mac/Linux
mv ~/Desktop/Files/best_readmission_model_xgboost.pkl models/
mv ~/Desktop/Files/meta_model.pkl models/
mv ~/Desktop/Files/preprocessor.pkl models/
mv ~/Desktop/Files/background_data.csv models/
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will take 5-10 minutes as it downloads PyTorch, Transformers, and other large packages.

### Step 5: Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Deploy to Streamlit Cloud (Free)

### Step 1: Prepare GitHub Repository

1. **Create a GitHub account** (if you don't have one): https://github.com/join

2. **Install Git** (if not installed):
   - Windows: https://git-scm.com/download/win
   - Mac: `brew install git`
   - Linux: `sudo apt-get install git`

3. **Initialize Git in your project folder:**

```bash
cd hospital-readmission-app
git init
git add .
git commit -m "Initial commit: Hospital readmission prediction app"
```

4. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `hospital-readmission-app`
   - Make it **Public** (required for free Streamlit hosting)
   - Don't initialize with README (we already have one)
   - Click "Create repository"

5. **Push your code to GitHub:**

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/hospital-readmission-app.git
git branch -M main
git push -u origin main
```

### Step 2: Set Up Git LFS (for model files)

Since the XGBoost model is 2.65 MB, we need Git LFS for proper handling:

```bash
# Install Git LFS
# Windows: Download from https://git-lfs.github.com/
# Mac: brew install git-lfs
# Linux: sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "models/*.pkl"
git lfs track "models/*.csv"

# Add and commit
git add .gitattributes
git add models/
git commit -m "Add model files with LFS"
git push
```

### Step 3: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**: https://streamlit.io/cloud

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Configure deployment:**
   - **Repository**: `YOUR_USERNAME/hospital-readmission-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom name (e.g., `hospital-readmission-predictor`)

5. **Click "Deploy"**

6. **Wait 5-10 minutes** for initial deployment (downloading models and dependencies)

7. **Your app will be live at**: `https://YOUR-APP-NAME.streamlit.app`

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

#  Troubleshooting

### Common Issues

**Issue 1: "ModuleNotFoundError"**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Issue 2: "BERT model download fails"**
```bash
# Solution: Check internet connection, model will download on first run
# Model size: ~440 MB from HuggingFace
```

**Issue 3: "File not found: models/..."**
```bash
# Solution: Ensure all 4 files are in the models/ folder
ls models/  # Mac/Linux
dir models\  # Windows
```

**Issue 4: Streamlit Cloud deployment fails**
- Ensure repository is **public**
- Check that `requirements.txt` is in the root directory
- Verify model files are committed with Git LFS
- Check Streamlit Cloud logs for specific errors

**Issue 5: App runs but predictions fail**
- Ensure all dropdown fields are selected (not "Select...")
- Ensure clinical notes text area is not empty
- Check that numerical inputs are valid (non-negative)

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

---

#  Security & Privacy

**Important Considerations:**

1. **No PHI Storage**: The app does NOT store any patient data
2. **Session-based**: All data is cleared when browser is closed
3. **HIPAA Compliance**: For production use in healthcare settings:
   - Deploy on private, secure servers
   - Implement authentication and authorization
   - Enable audit logging
   - Encrypt data in transit and at rest
4. **Clinical Validation**: Models should be validated on your institution's data before clinical use

---

## File Descriptions

| File | Size | Purpose |
|------|------|---------|
| `app.py` | ~20 KB | Main Streamlit application |
| `requirements.txt` | ~500 B | Python dependencies |
| `README.md` | ~10 KB | Documentation |
| `models/best_readmission_model_xgboost.pkl` | 2.65 MB | XGBoost trained model |
| `models/meta_model.pkl` | 911 B | Logistic regression meta-learner |
| `models/preprocessor.pkl` | ~50 KB | Feature preprocessing pipeline |
| `models/background_data.csv` | ~100 KB | SHAP background data |

**Total repository size**: ~3 MB (excluding BERT model which streams from HuggingFace)

---

## Customization

### Change Color Theme

Edit the CSS in `app.py` (lines 30-40):

```python
:root {
    --light-blue: #B5C7D2;      # Change to your primary color
    --dark-blue: #88A3B1;       # Change to your secondary color
    --darkest-blue: #467C9B;    # Change to your accent color
}
```

### Modify Risk Thresholds

Edit the `get_risk_label()` function in `app.py`:

```python
def get_risk_label(probability):
    if probability >= 0.6:      # Change threshold
        return "HIGH RISK", "risk-high"
    elif probability >= 0.4:    # Change threshold
        return "MEDIUM RISK", "risk-medium"
    else:
        return "LOW RISK", "risk-low"
```

### Add Hospital Logo

Add this code after the header in `app.py`:

```python
st.image("path/to/logo.png", width=200)
```

---

# Known Limitations

1. **First Load Time**: Initial model download takes 2-3 minutes
2. **Memory Usage**: BERT model requires ~1.5 GB RAM
3. **Processing Time**: Prediction takes 10-30 seconds depending on text length
4. **Browser Compatibility**: Best viewed in Chrome, Firefox, or Edge
5. **Mobile View**: Functional but desktop recommended for best experience

---

# Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check HuggingFace model card: https://huggingface.co/sujangauchan/DISCHARGY_SUMMARY_BERT_READMISSION_CLASSIFIER

---

# License

This application is for research and educational purposes. For clinical deployment, ensure proper validation and regulatory compliance.

---

# Deployment Checklist

Before deploying, ensure:

- [ ] All 4 model files are in `models/` folder
- [ ] `requirements.txt` is in root directory
- [ ] `app.py` is in root directory
- [ ] Git LFS is configured for model files
- [ ] Repository is pushed to GitHub
- [ ] Repository is set to **public**
- [ ] Streamlit Cloud account is created
- [ ] App is deployed and accessible

