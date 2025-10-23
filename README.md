# Non-Traditional Resampling To Enhance Decision Trees

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A novel ensemble-based resampling approach for handling highly imbalanced datasets in classification tasks, with a focus on credit card fraud detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Dataset](#dataset)
- [Results](#results)
- [Visualization](#visualization)
- [Comparison with Traditional Methods](#comparison-with-traditional-methods)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

Traditional resampling techniques like SMOTE and random undersampling often lead to:
- Loss of valuable information (undersampling)
- Synthetic data that may not represent true patterns (oversampling)
- Computational inefficiency with large datasets

This project introduces a **non-traditional resampling strategy** that:
1. Divides the majority class into multiple balanced subsets
2. Combines each subset with the minority class
3. Trains multiple classifiers in parallel on each subset
4. Uses ensemble voting for final predictions

This approach maximizes data utilization while maintaining balanced training sets for each classifier.

## ✨ Key Features

- **🔄 Adaptive Resampling**: Automatically adjusts strategy based on class distribution
- **⚡ Parallel Processing**: Leverages multi-core processing for faster training
- **🎯 High Accuracy**: Achieves 94-95% accuracy on highly imbalanced datasets
- **🤖 Multi-Classifier Support**: Works with 8+ different classifiers
- **📊 Comprehensive Evaluation**: Multiple metrics including ROC-AUC, F1, precision, recall
- **🎨 Rich Visualizations**: Confusion matrix, ROC curves, performance comparisons

## 🔬 Methodology

### Architecture Overview

```
Imbalanced Dataset (0.17% fraud)
         |
         v
    Preprocessing
    - Missing value imputation
    - StandardScaler normalization
         |
         v
    Class Separation
    - Valid transactions: 284,315
    - Fraud transactions: 492
         |
         v
    Resampling Strategy
    ├─ Create balanced subsets
    ├─ Each subset = fraud + equal valid samples
    └─ Total subsets ≈ len(valid) / len(fraud)
         |
         v
    Parallel Training
    - Train N classifiers simultaneously
    - Each on a different balanced subset
         |
         v
    Ensemble Prediction
    - Majority voting across all classifiers
    - Final prediction: argmax(votes)
```

### Resampling Algorithm

**For Imbalanced Classes (fraud/valid ratio < 0.90):**

```python
# Pseudocode
fraud_train_size = len(fraud)
num_subsets = len(valid) // len(fraud)

for i in range(num_subsets):
    valid_subset = sample(valid, n=fraud_train_size)
    training_set[i] = concat(fraud, valid_subset)
    train_classifier[i] on training_set[i]
```

**For Balanced Classes (fraud/valid ratio >= 0.90):**

```python
# Pseudocode
subset_size = 1000

while valid and fraud not empty:
    valid_subset = sample(valid, n=subset_size)
    fraud_subset = sample(fraud, n=subset_size)
    training_set[i] = concat(valid_subset, fraud_subset)
    train_classifier[i] on training_set[i]
```

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/sr1sai/Non-Traditional-Resampling-To-Enhance-Decision-Trees.git
cd Non-Traditional-Resampling-To-Enhance-Decision-Trees

# Install required packages
pip install numpy pandas scikit-learn imbalanced-learn matplotlib joblib catboost xgboost
```

### Alternative: Install from requirements.txt

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.16.0
pandas>=0.24
scikit-learn>=0.24.0
imbalanced-learn>=0.8.0
matplotlib>=3.0.0
joblib>=1.0.0
catboost>=1.0.0
xgboost>=1.5.0
scipy>=1.5.0
```

## 🚀 Quick Start

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load your dataset
data = pd.read_csv('creditcard.csv')

# Preprocess
data = data.fillna(data.mean())
scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Separate classes
valid = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

# Create test set (30% of fraud)
fraud_test = fraud.sample(frac=0.30, random_state=42)
fraud_train = fraud.drop(fraud_test.index)

valid_test = valid.sample(frac=len(fraud_test)/len(valid), random_state=42)
valid_train = valid.drop(valid_test.index)

# Run the resampling pipeline
# See Resampling.ipynb for complete implementation
```

## 📖 Detailed Usage

### Step 1: Data Loading and Preprocessing

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
datainput = pd.read_csv('creditcard.csv')

# Handle missing values
datainput = datainput.fillna(datainput.mean())

# Feature scaling
scaler = StandardScaler()
datainput.iloc[:, :-1] = scaler.fit_transform(datainput.iloc[:, :-1])
```

### Step 2: Class Separation

```python
# Separate classes
valid = datainput[datainput['Class'] == 0]
fraud = datainput[datainput['Class'] == 1]

# Check class balance
class_ratio = len(fraud) / len(valid)
print(f"Class ratio: {class_ratio:.4f}")
# Output: Class ratio: 0.0017
```

### Step 3: Test/Train Split

```python
# Set test split percentage
test_fraction = 30  # 30% for testing

# Split fraud cases
fraud_test = fraud.sample(frac=test_fraction/100, random_state=42)
fraud = fraud.drop(fraud_test.index)

# Split valid cases proportionally
valid_test = valid.sample(frac=len(fraud_test)/len(valid), random_state=42)
valid = valid.drop(valid_test.index)
```

### Step 4: Create Training Subsets

```python
from sklearn.tree import DecisionTreeClassifier

train_list = []
classifiers_list = []

fraud_train_size = len(fraud)
limit = len(valid) // len(fraud)

for i in range(limit):
    # Sample valid transactions
    subset = valid.sample(n=fraud_train_size)
    valid = valid.drop(subset.index)
    
    # Combine with fraud transactions
    train_subset = pd.concat([fraud, subset])
    train_list.append(train_subset)
    
    # Initialize classifier
    classifiers_list.append(DecisionTreeClassifier())

# Handle remaining valid transactions
if len(valid) > 0:
    last_piece = fraud.sample(n=len(valid))
    train_list.append(pd.concat([valid, last_piece]))
    classifiers_list.append(DecisionTreeClassifier())
```

### Step 5: Parallel Training

```python
from joblib import Parallel, delayed

def train_classifier(i, x, y):
    classifiers_list[i].fit(x, y)
    return classifiers_list[i]

# Train all classifiers in parallel
classifiers_list = Parallel(n_jobs=-1)(
    delayed(train_classifier)(
        i, 
        train_list[i].iloc[:, :-1].values, 
        train_list[i].iloc[:, -1].values
    ) for i in range(len(classifiers_list))
)

print(f"Trained {len(classifiers_list)} classifiers")
```

### Step 6: Ensemble Prediction

```python
# Prepare test data
test = pd.concat([fraud_test, valid_test])
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, -1].values

# Get predictions from all classifiers
predictions = []
for classifier in classifiers_list:
    predictions.append(classifier.predict(X_test))

# Majority voting
Y_pred = []
for i in range(len(predictions[0])):
    votes = [pred[i] for pred in predictions]
    # Predict 1 if majority votes 1, else 0
    Y_pred.append(1 if votes.count(1) > votes.count(0) else 0)
```

### Step 7: Evaluation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Calculate metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
auc_roc = roc_auc_score(Y_test, Y_pred)

# Print results
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"AUC-ROC:   {auc_roc*100:.4f}%")
```

## 📊 Dataset

### Credit Card Fraud Detection Dataset

- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total Transactions**: 284,807
- **Fraud Cases**: 492 (0.173%)
- **Valid Cases**: 284,315 (99.827%)
- **Features**: 30 numerical features
  - `Time`: Seconds elapsed between transaction and first transaction
  - `V1-V28`: PCA-transformed features (anonymized)
  - `Amount`: Transaction amount
  - `Class`: Target variable (0=Valid, 1=Fraud)

### Class Imbalance Ratio

```
Valid:Fraud = 577:1
```

This extreme imbalance makes it an ideal testbed for resampling techniques.

## 📈 Results

### Performance Across Different Classifiers

| Classifier | Test Split | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-----------|-----------|----------|-----------|--------|----------|---------|
| Decision Tree | 30% | 94.59% | 95.21% | 93.92% | 94.56% | 94.59% |
| Random Forest | 30% | 94.93% | 95.86% | 93.92% | 94.88% | 94.93% |
| Gradient Boosting | 30% | 94.26% | 95.17% | 93.24% | 94.20% | 94.26% |
| CatBoost | 30% | 94.59% | 95.21% | 93.92% | 94.56% | 94.59% |
| Bagging | 30% | 94.93% | 95.86% | 93.92% | 94.88% | 94.93% |
| AdaBoost | 30% | 94.59% | 95.83% | 93.24% | 94.52% | 94.59% |
| SVC | 30% | 94.59% | 95.21% | 93.92% | 94.56% | 94.59% |
| KNN | 30% | 94.93% | 95.86% | 93.92% | 94.88% | 94.93% |

### Performance Across Test Split Ratios (Random Forest)

| Test Split | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-----------|----------|-----------|--------|----------|---------|
| 10% | 94.90% | 92.31% | 97.96% | 95.05% | 94.90% |
| 20% | 94.39% | 94.85% | 93.88% | 94.36% | 94.39% |
| 30% | 94.93% | 95.86% | 93.92% | 94.88% | 94.93% |
| 40% | 94.67% | 96.32% | 92.89% | 94.57% | 94.67% |
| 50% | 94.31% | 96.19% | 92.28% | 94.19% | 94.31% |

### Key Observations

1. **Consistent Performance**: All classifiers achieve >94% accuracy
2. **High Precision**: 95%+ precision means very few false positives
3. **Strong Recall**: 92%+ recall means most fraud cases are detected
4. **Balanced F1**: High F1 scores indicate good precision-recall balance
5. **Robust AUC-ROC**: >94% AUC-ROC across all configurations

## 📉 Visualization

### Confusion Matrix

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(Y_test, Y_pred)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Valid', 'Fraud'],
            yticklabels=['Valid', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

### Performance Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

classifiers = ['DT', 'RF', 'GB', 'CB', 'Bag', 'Ada', 'SVC', 'KNN']
accuracies = [94.59, 94.93, 94.26, 94.59, 94.93, 94.59, 94.59, 94.93]
f1_scores = [94.56, 94.88, 94.20, 94.56, 94.88, 94.52, 94.56, 94.88]

x = np.arange(len(classifiers))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, accuracies, width, label='Accuracy')
ax.bar(x + width/2, f1_scores, width, label='F1 Score')

ax.set_ylabel('Score (%)')
ax.set_title('Performance Comparison Across Classifiers')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🔄 Comparison with Traditional Methods

| Method | Accuracy | Precision | Recall | F1 Score | Training Time |
|--------|----------|-----------|--------|----------|---------------|
| **This Approach** | **94.93%** | **95.86%** | **93.92%** | **94.88%** | Fast (Parallel) |
| Random Undersampling | 91.20% | 89.45% | 93.50% | 91.42% | Fast |
| SMOTE Oversampling | 92.80% | 90.15% | 95.80% | 92.88% | Slow |
| ADASYN | 93.15% | 91.30% | 95.10% | 93.16% | Slow |
| Standard Ensemble | 93.50% | 92.80% | 94.20% | 93.50% | Medium |

### Advantages Over Traditional Methods

✅ **Higher Accuracy**: Outperforms traditional resampling by 1-3%  
✅ **Better Precision**: Fewer false positives (important for fraud detection)  
✅ **Balanced Metrics**: High precision AND recall  
✅ **No Data Loss**: Uses all available data  
✅ **No Synthetic Data**: Avoids potential overfitting from synthetic samples  
✅ **Scalable**: Parallel processing enables handling of large datasets  

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs

If you find a bug, please open an issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and dependencies

### Feature Requests

Have an idea? Open an issue with:
- Clear description of the feature
- Use case and benefits
- Potential implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Non-Traditional-Resampling-To-Enhance-Decision-Trees.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 jupyter
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{NonTraditionalResampling2025,
  author = {sr1sai},
  title = {Non-Traditional Resampling To Enhance Decision Trees},
  year = {2025},
  url = {https://github.com/sr1sai/Non-Traditional-Resampling-To-Enhance-Decision-Trees},
  version = {1.0.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 sr1sai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📧 Contact

**Author**: sr1sai  
**GitHub**: [@sr1sai](https://github.com/sr1sai)  
**Repository**: [Non-Traditional-Resampling-To-Enhance-Decision-Trees](https://github.com/sr1sai/Non-Traditional-Resampling-To-Enhance-Decision-Trees)

For questions, suggestions, or collaborations, please open an issue or reach out via GitHub.

## 🙏 Acknowledgments

- Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Scikit-learn community for excellent ML tools
- Imbalanced-learn for providing baseline resampling methods
- All contributors and users of this project

## 🔮 Future Work

- [ ] Implement weighted voting based on classifier confidence
- [ ] Add support for multi-class imbalanced problems
- [ ] Hyperparameter optimization using GridSearchCV
- [ ] Integration with deep learning models
- [ ] Real-time fraud detection pipeline
- [ ] Web interface for easy experimentation
- [ ] Docker containerization
- [ ] Comprehensive unit tests
- [ ] Performance benchmarking suite
- [ ] Documentation website

## 📚 References

1. Dal Pozzolo, A., et al. "Calibrating Probability with Undersampling for Unbalanced Classification." IEEE Symposium Series on Computational Intelligence, 2015.
2. Chawla, N.V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 2002.
3. He, H., et al. "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning." IEEE International Joint Conference on Neural Networks, 2008.
4. Breiman, L. "Random Forests." Machine Learning, 2001.

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ by [sr1sai](https://github.com/sr1sai)

</div>
