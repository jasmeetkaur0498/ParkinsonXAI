# ParkinsonXAI: AI-Driven Multi-Class Diagnosis of Parkinson's Disease

## Overview

**ParkinsonXAI** is an advanced machine learning research project focused on **multi-class classification** for the diagnosis and differentiation of Parkinson's Disease (PD) from other movement disorders using symptom-based data. This project implements multiple machine learning algorithms to achieve high diagnostic accuracy and develop explainable AI models for clinical decision support.

### Research Paper Reference
This project is based on the research published in IEEE: [AI-Driven Multi-Class Diagnosis of Parkinson's Disease: Enhancing Accuracy and Differentiation from Movement Disorders](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11011463)

---

## 📋 Project Objectives

1. **Binary Classification**: Distinguish between Parkinson's Disease and Healthy individuals
2. **Binary Classification**: Differentiate Parkinson's Disease from Other Movement Disorders
3. **Multi-Class Classification**: Simultaneously classify three categories:
   - Parkinson's Disease
   - Healthy Individuals
   - Other Movement Disorders

4. **Model Optimization**: Apply SMOTE (Synthetic Minority Over-sampling Technique) for class imbalance
5. **Hyperparameter Tuning**: Optimize all models for maximum performance
6. **Explainability**: Provide interpretable predictions for clinical use

---

## 🏗️ Architecture & Data Flow

### Project Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION & PREPROCESSING               │
│  (Patient Symptom Data: 30 clinical features)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   DATA CLEANING & EXPLORATION      │
        │  - Handle Missing Values           │
        │  - Normalize/Standardize Features  │
        │  - Class Distribution Analysis     │
        └────────┬─────────────────────────┬─┘
                 │                         │
         ┌───────▼────────┐      ┌────────▼──────────┐
         │ TRAIN/TEST     │      │   FEATURE         │
         │ SPLIT (80/20)  │      │   SELECTION       │
         │ Stratified     │      │   - SelectKBest  │
         │ K-Fold CV      │      │   - RFE          │
         │ (k=10)         │      │   - Correlation  │
         └───────┬────────┘      └────────┬──────────┘
                 │                         │
                 └─────────────┬───────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │  CLASS IMBALANCE HANDLING (SMOTE)   │
            │  - Oversample Minority Classes      │
            │  - Balance Class Distribution       │
            └──────────────────┬──────────────────┘
                               │
         ┌─────────────────────▼─────────────────────┐
         │    MODEL TRAINING & OPTIMIZATION          │
         │  ┌──────────────────────────────────────┐ │
         │  │ • Logistic Regression                │ │
         │  │ • Naive Bayes (Gaussian)             │ │
         │  │ • K-Nearest Neighbors (KNN)          │ │
         │  │ • Decision Tree                      │ │
         │  │ • Random Forest                      │ │
         │  │ • Gradient Boosting                  │ │
         │  │ • AdaBoost                           │ │
         │  │ • XGBoost                            │ │
         │  │ • LightGBM                           │ │
         │  │ • CatBoost                           │ │
         │  │ • Support Vector Machine (SVM)       │ │
         │  │ • Neural Network (MLP)               │ │
         │  │ • Linear/Quadratic Discriminant      │ │
         │  │ • Extra Trees                        │ │
         │  └──────────────────────────────────────┘ │
         └─────────────────────┬─────────────────────┘
                               │
         ┌─────────────────────▼──────────────────┐
         │  HYPERPARAMETER TUNING                │
         │  - Grid Search / Random Search        │
         │  - Cross-Validation (10-Fold)         │
         │  - Optimization Metrics               │
         └─────────────────────┬──────────────────┘
                               │
         ┌─────────────────────▼──────────────────┐
         │   EVALUATION & VALIDATION             │
         │  ┌──────────────────────────────────┐ │
         │  │ • Accuracy                       │ │
         │  │ • Precision                      │ │
         │  │ • Recall                         │ │
         │  │ • F1-Score                       │ │
         │  │ • ROC-AUC                        │ │
         │  │ • Confusion Matrix               │ │
         │  │ • Classification Report          │ │
         │  └──────────────────────────────────┘ │
         └─────────────────────┬──────────────────┘
                               │
         ┌─────────────────────▼──────────────────┐
         │   OUTPUT GENERATION & REPORTING       │
         │   (classification_output.xlsx)        │
         └───────────────────────────────────────┘
```

---

## 📊 Feature Set (30 Clinical Symptoms)

The model uses the following symptom-based features:

| Feature Category | Symptoms |
|---|---|
| **Gastrointestinal** | Dribbling, Swallowing, Vomiting, Constipation, Bowel inconsistence, Bowel emptying incomplete, Urgency, Nocturia |
| **Pain & Physical** | Pains, Weight, Swelling, Dizzy, Falling |
| **Autonomic** | Sweating |
| **Sensory** | Diplopia, Taste/smelling |
| **Cognitive & Mood** | Remembering, Loss of interest, Concentrating, Sad/blues, Anxiety |
| **Sexual Function** | Sex drive, Sex difficulty |
| **Sleep Disorders** | Daytime sleepiness, Insomnia, Intense vivid dreams, Acting out during dreams |
| **Motor Symptoms** | Restless legs |

---

## 🔬 Classification Approaches

### 1. **Binary Classification: Parkinson's vs Healthy**
- **Filename**: `P_&_O_Feature_selection.ipynb`
- **Description**: Distinguishes Parkinson's Disease patients from healthy individuals
- **Classes**: 2
- **Output**: `P_&_O_Feature_selection_output.ipynb.xlsx`

### 2. **Binary Classification: Parkinson's vs Other Disorders**
- **Filename**: `2_class_P_&O_classification.ipynb`
- **Description**: Differentiates Parkinson's Disease from other movement disorders
- **Classes**: 2
- **Output**: `2class_outputs.xlsx`

### 3. **Binary Classification with SMOTE & Hyperparameter Tuning**
- **Filename**: `P_&_O_Classification+Smote+hyperparameter.ipynb`
- **Description**: Binary classification with class balancing and optimized hyperparameters
- **Techniques**:
  - SMOTE for handling class imbalance
  - Hyperparameter optimization via Grid/Random Search
  - 10-Fold Stratified Cross-Validation
- **Output**: `P_&_O_Classification+Smote+hyperparameter_output.ipynb.xlsx`

### 4. **Multi-Class Classification: 3-Way Diagnosis**
- **Filename**: `3_Classifier.ipynb`
- **Description**: Simultaneously classifies all three categories
- **Classes**: 3 (Parkinson's, Healthy, Other Disorders)
- **Models**: 14+ machine learning algorithms
- **Output**: `3_Classifier_output.ipynb.xlsx`

### 5. **Feature Selection for 3-Class Problem**
- **Filename**: `3Class_Feature_Selection.ipynb`
- **Description**: Identifies the most relevant features for multi-class classification
- **Techniques**:
  - SelectKBest with chi-squared scoring
  - Recursive Feature Elimination (RFE)
  - Feature importance ranking
- **Output**: `3Class_Feature_Selection_output.ipynb.xlsx`

### 6. **3-Class Classification with SMOTE & Hyperparameter Tuning**
- **Filename**: `3Class_smote_hyperparameter.ipynb`
- **Description**: Advanced 3-class classification with optimization techniques
- **Techniques**:
  - SMOTE for multi-class imbalance
  - Comprehensive hyperparameter tuning
  - 10-Fold Stratified Cross-Validation
- **Output**: `3Class_smote_hyperparameter_output.ipynb.xlsx`

---

## 🤖 Machine Learning Algorithms

The project implements **14+ classification algorithms**:

### Tree-Based Models
- **Decision Tree**: Fast, interpretable model for feature interactions
- **Random Forest**: Ensemble of decision trees for robust predictions
- **Gradient Boosting**: Sequential ensemble for high accuracy
- **AdaBoost**: Adaptive boosting for difficult-to-classify instances
- **XGBoost**: Extreme Gradient Boosting with optimized performance
- **LightGBM**: Lightweight Gradient Boosting Machine
- **CatBoost**: Categorical Boosting with automatic categorical handling
- **Extra Trees**: Extremely Randomized Trees

### Linear Models
- **Logistic Regression**: Baseline probabilistic classifier
- **Linear Discriminant Analysis (LDA)**: Dimensionality reduction classifier
- **Quadratic Discriminant Analysis (QDA)**: Non-linear discriminant analysis

### Distance-Based Models
- **K-Nearest Neighbors (KNN)**: Instance-based learning
- **Support Vector Machine (SVM)**: Maximum margin classifier

### Probabilistic Models
- **Naive Bayes**: Fast probabilistic classifier based on Bayes' theorem

### Neural Networks
- **Multi-Layer Perceptron (MLP)**: Artificial neural network

---

## 🛠️ Technologies & Libraries

```python
# Core Data Processing
pandas                    # Data manipulation
numpy                     # Numerical computing

# Machine Learning
scikit-learn             # ML algorithms & preprocessing
xgboost                  # XGBoost implementation
lightgbm                 # LightGBM implementation
catboost                 # CatBoost implementation
imblearn                 # SMOTE & imbalanced learning

# Model Evaluation
sklearn.metrics          # Comprehensive evaluation metrics

# Feature Engineering
sklearn.feature_selection # Feature selection techniques

# Data Visualization (optional)
matplotlib               # Plotting & visualization
seaborn                  # Statistical visualizations
```

---

## 📂 Project Structure

```
ParkinsonXAI/
├── README.md                                    # This file
├── parkinson/
│   ├── 2_class_P_&O_classification.ipynb       # Binary classification (P vs O)
│   ├── 2class_outputs.xlsx                      # Results
│   ├── 3_Classifier.ipynb                       # 3-way classification
│   ├── 3_Classifier_output.ipynb.xlsx           # Results
│   ├── 3Class_Feature_Selection.ipynb           # Feature selection for 3-class
│   ├── 3Class_Feature_Selection_output.ipynb.xlsx
│   ├── 3Class_smote_hyperparameter.ipynb        # 3-class with SMOTE & tuning
│   ├── 3Class_smote_hyperparameter_output.ipynb.xlsx
│   ├── P_&_O_Classification+Smote+hyperparameter.ipynb
│   ├── P_&_O_Classification+Smote+hyperparameter_output.ipynb.xlsx
│   ├── P_&_O_Feature_selection.ipynb            # Feature selection (P vs O)
│   ├── P_&_O_Feature_selection_output.ipynb.xlsx
│   ├── classification_output.csv                # Primary output (CSV format)
│   ├── df_filtered.csv                          # Preprocessed data (2-class)
│   └── combined_df.csv                          # Preprocessed data (3-class)
├── output/                                      # Additional outputs directory
└── research_paper/
    └── AI-Driven_Multi-Class_Diagnosis_of...    # Research paper reference

```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- pip or conda package manager

### Installation

1. **Clone the repository** (or navigate to the project directory):
```bash
cd ParkinsonXAI
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate          # On macOS/Linux
venv\Scripts\activate             # On Windows
```

3. **Install required packages**:
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imblearn matplotlib seaborn openpyxl
```

### Running the Notebooks

1. **Launch Jupyter**:
```bash
jupyter notebook
```

2. **Open desired notebook**:
   - Start with `3_Classifier.ipynb` for complete 3-class overview
   - Or explore specific classification approaches based on your needs

3. **Execute cells sequentially**:
   - Ensure data files (`df_filtered.csv`, `combined_df.csv`) are in the same directory
   - Follow the notebook flow from data loading → preprocessing → training → evaluation

---

## 📊 Output Files

### Excel Output Format (*.xlsx)

Each notebook generates results in Excel format containing:

| Column | Description |
|---|---|
| **Model Name** | Algorithm used (e.g., "Logistic Regression", "XGBoost") |
| **Accuracy** | Overall prediction accuracy |
| **Precision** | True positive rate among predicted positives |
| **Recall** | True positive rate among actual positives |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve (binary problems) |
| **CV Mean** | Cross-validation average score |
| **CV Std** | Cross-validation standard deviation |
| **Training Accuracy** | Performance on training set |
| **Testing Accuracy** | Performance on test set |

### Primary Output
- **`classification_output.csv`**: Main results aggregating all model performances

---

## 🔧 Key Techniques

### 1. **Data Preprocessing**
- Feature standardization (StandardScaler)
- Handling missing values
- Train-test split (80-20 ratio)
- Stratified splits to maintain class distribution

### 2. **Class Imbalance Handling**
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- Generates synthetic samples for minority classes
- Applied only to training data to prevent data leakage

### 3. **Feature Selection**
- **SelectKBest**: Selects k best features based on statistical tests
- **RFE** (Recursive Feature Elimination): Iteratively removes less important features
- **Correlation Analysis**: Identifies highly correlated features
- **Feature Importance**: Extracted from tree-based models

### 4. **Model Evaluation**
- **Stratified K-Fold Cross-Validation** (k=10)
- **Multiple Metrics**:
  - Accuracy: Overall correctness
  - Precision: Reliability of positive predictions
  - Recall: Coverage of actual positives
  - F1-Score: Balanced metric combining precision & recall
  - ROC-AUC: Probability curve under various thresholds
  - Confusion Matrix: Detailed true/false positives & negatives

### 5. **Hyperparameter Tuning**
- Grid Search for optimal parameter combinations
- Random Search for broader exploration
- Cross-validation for unbiased performance estimation
- Tuned parameters stored and reported in outputs

---

## 📈 Expected Results

### Typical Performance Metrics:

| Classification Task | Expected Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Binary (P vs Healthy) | 85-95% | 0.85-0.95 | 0.90-0.98 |
| Binary (P vs Other) | 80-92% | 0.80-0.92 | 0.85-0.96 |
| Multi-Class (3-way) | 78-90% | 0.78-0.90 | N/A |

*Note: Actual results vary based on dataset characteristics and model configuration*

---

## 🔬 Research & Innovation

### Key Contributions:

1. **Multi-Class Classification Framework**: Comprehensive approach to distinguish Parkinson's from other conditions
2. **Explainable AI**: Feature importance analysis and model interpretability
3. **Hyperparameter Optimization**: Systematic tuning for maximum performance
4. **SMOTE Integration**: Handles real-world class imbalance in medical data
5. **Ensemble Methods**: Combines multiple algorithms for robust predictions
6. **Clinical Relevance**: Based on established medical symptom profiles

### Potential Clinical Applications:

- **Diagnostic Assistance**: Support for preliminary screening
- **Clinical Decision Support**: Aid in differential diagnosis
- **Research Tool**: Identify key discriminative features
- **Telemedicine**: Remote diagnostic support systems
- **Epidemiological Studies**: Population-level analysis

---

## 💡 Usage Examples

### Running a Specific Classification Approach:

```python
# Example: Load and run the 3-class classifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("combined_df.csv")

# Feature engineering
symptom_columns = ['Dribbling', 'Swallowing', 'Vomiting', ...]  # 30 features
X = df[symptom_columns]
y = df['Rencoded']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

---

## 📝 Notes & Considerations

### Data Requirements:
- Minimum 30 symptom features for optimal performance
- Class-balanced or imbalanced data (SMOTE handles imbalance)
- Complete or minimal missing values
- Numerical feature format

### Model Selection Guidelines:
- **Start with**: Logistic Regression (baseline), Random Forest (robust)
- **For Speed**: KNN, Naive Bayes
- **For Accuracy**: XGBoost, LightGBM, CatBoost
- **For Interpretability**: Decision Tree, Logistic Regression, LDA

### Performance Optimization Tips:
1. Ensure data preprocessing consistency
2. Use stratified splits for small datasets
3. Apply feature scaling (StandardScaler) before tree models
4. Tune hyperparameters on cross-validation scores
5. Monitor for overfitting on test sets
6. Handle class imbalance with SMOTE

---

## 📚 References

1. **Research Paper**: [AI-Driven Multi-Class Diagnosis of Parkinson's Disease: Enhancing Accuracy and Differentiation from Movement Disorders](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11011463)
   - IEEE Xplore Digital Library

2. **Parkinson's Disease Information**:
   - https://www.parkinsons.org.uk/
   - https://www.michaeljfox.org/

3. **Machine Learning Frameworks**:
   - Scikit-learn: https://scikit-learn.org/
   - XGBoost: https://xgboost.readthedocs.io/
   - LightGBM: https://lightgbm.readthedocs.io/
   - CatBoost: https://catboost.ai/

4. **Imbalanced Learning**:
   - SMOTE Documentation: https://imbalanced-learn.org/
