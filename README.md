# Phishing Detection using Machine Learning and Deep Learning

A comprehensive comparative study of traditional machine learning and deep learning approaches for phishing website detection, with a focus on African deployment contexts.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Academic Report](#academic-report)
- [Citation](#citation)
- [License](#license)

---

## 🔍 Overview

This project implements and compares multiple machine learning and deep learning models for detecting phishing websites based on URL and webpage features. The study evaluates 5 distinct approaches:

1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble decision tree method
3. **XGBoost** - Gradient boosting framework
4. **Sequential MLP** - Feedforward neural network (Sequential API)
5. **Functional MLP** - Feedforward neural network (Functional API)

The research emphasizes practical deployment considerations for resource-constrained environments, particularly in African contexts where infrastructure limitations and computational resources require efficient solutions.

---

## 📊 Dataset

### Source
**Phishing Websites Dataset**
UCI Machine Learning Repository
🔗 https://archive.ics.uci.edu/dataset/327/phishing+websites

### Description
- **Total Samples**: 11,055 instances
- **Features**: 30 attributes
- **Classes**: Binary (Phishing / Legitimate)
- **Format**: CSV

### Features Include:
- **URL-based features**: Length, special characters, IP address presence
- **Domain-based features**: Domain age, DNS records, WHOIS information
- **Content-based features**: Page rank, SSL certificate properties
- **Statistical features**: Request URL patterns, anchor URL characteristics

### Data Splits:
- **Training Set**: 60% (6,633 samples)
- **Validation Set**: 20% (2,211 samples)
- **Test Set**: 20% (2,211 samples)

All splits use stratified sampling to maintain class distribution.

---

## 🤖 Models Implemented

### 1. Logistic Regression
- **Type**: Linear classifier
- **Configuration**: Balanced class weights, L2 regularization (C=1.0)
- **Purpose**: Baseline performance benchmark
- **Test AUC**: 0.9804

### 2. Random Forest
- **Type**: Ensemble method
- **Configuration**: 200 trees, unlimited depth
- **Advantages**: High interpretability, feature importance scores
- **Test AUC**: 0.9955

### 3. XGBoost (Best Performer)
- **Type**: Gradient boosting
- **Configuration**: Tree-based boosting with regularization
- **Advantages**: State-of-the-art performance, efficient training
- **Test AUC**: 0.9963 ⭐

### 4. Sequential MLP
- **Type**: Deep neural network
- **Architecture**: Multiple fully connected layers with ReLU activation
- **Regularization**: Dropout, batch normalization
- **Test AUC**: 0.9944

### 5. Functional MLP
- **Type**: Deep neural network (Functional API)
- **Architecture**: Alternative implementation for flexibility
- **Regularization**: Dropout, batch normalization
- **Test AUC**: 0.9954

---

## 📈 Results

### Performance Summary

| Model | Test AUC | Test Precision | Test Recall | Test F1 | Rank |
|-------|----------|----------------|-------------|---------|------|
| **XGBoost** | **0.9963** | 0.9648 | 0.9784 | 0.9715 | **1st** ⭐ |
| Random Forest | 0.9955 | 0.9680 | 0.9827 | 0.9753 | 2nd |
| Functional MLP | 0.9954 | 0.9616 | 0.9751 | 0.9683 | 3rd |
| Sequential MLP | 0.9944 | 0.9633 | 0.9654 | 0.9643 | 4th |
| Logistic Regression | 0.9804 | 0.9295 | 0.9416 | 0.9355 | 5th |

### Key Metrics:
- **Best Overall Performance**: XGBoost (0.9963 AUC)
- **Best F1-Score**: Random Forest (0.9753)
- **Best Recall**: Random Forest (0.9827)
- **Fastest Training**: Logistic Regression (<1 minute)
- **Best Error Rate**: XGBoost (4.5% FPR, 2.2% FNR)

### Generalization:
All models demonstrate excellent generalization with negative validation-test gaps, indicating robust performance on unseen data without overfitting.

---

## 📁 Repository Structure

```
final/
├── phishing_ml_vs_dl_notebook_enhanced.ipynb   # Main Jupyter notebook
├── README.md                                    # This file
├── experiment_results.csv                       # Model performance results
├── phishing.csv                                 # Full dataset
├── train_split.csv                              # Training data (60%)
├── val_split.csv                                # Validation data (20%)
├── test_split.csv                               # Test data (20%)
│
├── Academic Report/
│   ├── Phishing_Detection_Africa_Report_FINAL.docx    # Word format
│   ├── Phishing_Detection_Africa_Report_FINAL.pdf     # PDF format
│   └── Phishing_Detection_Africa_Report_CORRECTED.md  # Markdown source
│
├── Figures (Generated)/
│   ├── report_table1_actual.png                 # Performance summary table
│   ├── report_figure1_auc_actual.png            # AUC comparison chart
│   ├── report_figure2_metrics_actual.png        # Multi-metric comparison
│   └── report_figure3_generalization_actual.png # Generalization analysis
│
├── Figures (From Notebook)/
│   ├── notebook_image_4_cell36.png              # ROC/PR curves (XGBoost)
│   ├── notebook_image_5_cell36.png              # ROC/PR curves (MLP)
│   ├── notebook_image_6_cell39.png              # Learning curves
│   └── notebook_image_7_cell41.png              # Confusion matrices
│
└── Documentation/
    ├── FINAL_REPORT_WITH_CURVES.md              # Report documentation
    ├── REPORT_SUMMARY.md                        # Quick summary
    └── README_Report_Guide.md                   # Report guide
```

---

## 🛠️ Requirements

### Python Version
- Python 3.8+

### Core Libraries
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
keras>=2.8.0
```

### Visualization
```
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
```

### Utilities
```
jupyter>=1.0.0
notebook>=6.4.0
tqdm>=4.62.0
```

### Optional (for report generation)
```
python-docx>=0.8.11
docx2pdf>=0.1.8
markdown>=3.3.0
```

---

## 💻 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd final
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas scikit-learn xgboost tensorflow matplotlib seaborn jupyter
```

### 4. Download Dataset
The dataset is included as `phishing.csv`. If needed, download fresh copy from:
https://archive.ics.uci.edu/dataset/327/phishing+websites

---

## 🚀 Usage

### Running the Notebook

1. **Start Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open the notebook**
```
phishing_ml_vs_dl_notebook_enhanced.ipynb
```

3. **Run all cells sequentially**
   - Cell 1-5: Data loading and preprocessing
   - Cell 6-10: Exploratory Data Analysis (EDA)
   - Cell 11-20: Model training (Traditional ML)
   - Cell 21-35: Model training (Deep Learning)
   - Cell 36-40: Performance evaluation
   - Cell 41-50: Visualization and analysis

### Quick Start (Python Script)
```python
import pandas as pd
from sklearn.ensemble import RandomForest
from xgboost import XGBClassifier

# Load preprocessed data
train = pd.read_csv('train_split.csv')
test = pd.read_csv('test_split.csv')

X_train = train.drop('Result', axis=1)
y_train = train['Result']
X_test = test.drop('Result', axis=1)
y_test = test['Result']

# Train XGBoost (best model)
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import roc_auc_score
predictions = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, predictions)
print(f"Test AUC: {auc:.4f}")
```

### Running Individual Experiments

**Experiment 1: Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000)
lr.fit(X_train, y_train)
```

**Experiment 2: Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
```

**Experiment 3: XGBoost**
```python
import xgboost as xgb
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
```

---

## 🔑 Key Findings

### 1. Model Performance
- **XGBoost achieves best overall performance** (0.9963 AUC)
- Random Forest offers best F1-score and recall
- Deep learning competitive but not superior for tabular data
- All models exceed 0.98 AUC (excellent discrimination)

### 2. Generalization
- All models show excellent generalization (negative val-test gaps)
- No evidence of overfitting
- Robust performance on unseen data

### 3. Computational Efficiency
- Traditional ML models train in minutes
- Deep learning requires longer training (epochs monitoring)
- Inference times suitable for real-time deployment (<10ms)

### 4. Feature Importance (XGBoost)
Top contributing features:
1. SSL certificate age and validity
2. Domain registration length
3. Presence of IP address in URL
4. Abnormal URL structure indicators
5. Redirect count

### 5. Error Analysis
- **XGBoost**: 33 false positives, 20 false negatives
- **Random Forest**: 38 false positives, 16 false negatives
- **Logistic Regression**: 66 false positives, 54 false negatives
- XGBoost reduces errors by 50-62% vs baseline

### 6. African Context Recommendations
- **Financial Institutions**: XGBoost or Random Forest
- **Telecommunications**: Random Forest (high precision)
- **Educational Institutions**: Random Forest (interpretable)
- **Government Agencies**: XGBoost (maximum accuracy)
- **SMEs**: Random Forest (low maintenance, cost-effective)

---

## 📄 Academic Report

A comprehensive academic report (6,500+ words) is included documenting:

- Literature review (21 scholarly sources)
- Detailed methodology
- Experimental results with 8 figures
- African deployment context analysis
- Practical recommendations

**Files:**
- `Phishing_Detection_Africa_Report_FINAL.docx` - Word format
- `Phishing_Detection_Africa_Report_FINAL.pdf` - PDF format

**Visualizations included:**
- Performance comparison tables
- ROC and Precision-Recall curves
- Learning curves (loss and AUC over epochs)
- Confusion matrices
- Generalization analysis

---

## 🎓 Citation

If you use this work, please cite:

### Dataset Citation
```bibtex
@misc{uci_phishing_2015,
  author = {Rami M. Mohammad and Fadi Thabtah and Lee McCluskey},
  title = {Phishing Websites Dataset},
  year = {2015},
  howpublished = {UCI Machine Learning Repository},
  url = {https://archive.ics.uci.edu/dataset/327/phishing+websites}
}
```

### This Work
```bibtex
@techreport{phishing_africa_2025,
  title = {Machine Learning and Deep Learning Approaches for Phishing Detection:
           A Comparative Study with Focus on African Context},
  author = {[Your Name]},
  year = {2025},
  institution = {[Your Institution]},
  type = {Technical Report}
}
```

---

## 📚 References

Key references used in this study:

1. Mohammad, R. M., Thabtah, F., & McCluskey, L. (2014). Intelligent rule-based phishing websites classification. IET Information Security, 8(3), 153-160.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794).

3. GSMA (2023). The Mobile Economy: Sub-Saharan Africa 2023. GSM Association.

4. Sahingoz, O. K., Buber, E., Demir, O., & Diri, B. (2019). Machine learning based phishing detection from URLs. Expert Systems with Applications, 117, 345-357.

Full bibliography available in the academic report.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more deep learning architectures (CNN for URLs, Transformers)
- [ ] Implement ensemble voting mechanisms
- [ ] Add adversarial robustness testing
- [ ] Collect Africa-specific phishing dataset
- [ ] Add real-time deployment pipeline
- [ ] Implement explainability methods (SHAP, LIME)

---

## 📧 Contact

For questions, suggestions, or collaboration:

- **Email**: wadotiwawil@gmail.com
- **Institution**: African Leadership University

---

## 🔒 License

This project is licensed under the MIT License - see the LICENSE file for details.

### Dataset License
The Phishing Websites Dataset is available from the UCI Machine Learning Repository under their terms of use. Please review the dataset license before commercial use.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the phishing dataset
- African cybersecurity research community for context and insights
- Open-source ML/DL libraries: scikit-learn, XGBoost, TensorFlow
- Jupyter community for excellent notebook environment

---

## 📊 Quick Stats

- **Models**: 5 different approaches
- **Dataset Size**: 11,055 URLs
- **Features**: 30 attributes
- **Best AUC**: 0.9963 (XGBoost)
- **Training Time**: 2-20 minutes (depending on model)
- **Inference Speed**: <10 milliseconds per URL
- **Code Quality**: Fully documented and reproducible
- **Report**: 6,500+ words with 8 figures

---

## 🎯 Future Work

1. **Dataset Expansion**
   - Collect Africa-specific phishing examples
   - Include mobile phishing (SMS, WhatsApp)
   - Multilingual phishing detection

2. **Model Enhancement**
   - Transformer-based architectures
   - Graph neural networks for URL structure
   - Online learning for real-time adaptation

3. **Deployment**
   - Browser extension implementation
   - Mobile SDK for Android/iOS
   - Cloud API service
   - Edge deployment optimization

4. **Evaluation**
   - Adversarial robustness testing
   - Longitudinal performance tracking
   - Cross-dataset generalization
   - Real-world deployment study

---

**Last Updated**: 2025-01-18
**Version**: 1.0.0
**Status**: Production Ready ✅

---

For detailed methodology, results analysis, and African context discussion, please refer to the academic report included in this repository.

**Happy Phishing Detection! 🛡️**
