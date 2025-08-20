# 🏥 Insurance Premium Prediction (Annual Price)

## 📂 Repository Outline

```
    p1-ftds029-hck-m2-Khalif-Coding 
    ├── deployment/
    │   ├── app.py
    │   └── eda.py
    │   └── predict.py
    │   └── about.py
    │   └── Medicalpremium.pcsv
    │   └── requirement.txt  
    │   └── BestRFR.pkl
    │   └── utils.py
    ├── P1M2_Khalif.ipynb
    ├── P1M2_Khalif_Inference.ipynb
    └── Medicalpremium.pcsv
    └── README.md
    └── BestRFR.pkl
    └── utils.py
```


---

## 📌 Problem Background
Healthcare costs in the United States are extremely high. A single illness such as **cancer or heart surgery** can exceed **$100,000**, forcing families to drain savings, sell property, or even go bankrupt.  

Insurance helps mitigate these risks by covering most of the medical expenses. However, **insurance premiums** are determined by personal factors such as **age, BMI, smoking habits, and number of dependents**.  

This project aims to **predict annual insurance premiums** based on these personal factors, so families can estimate future costs and prepare for unexpected medical bills.  

---

## 🎯 Project Output
- A machine learning model that **predicts annual insurance premium costs**  
- Insights into which personal factors most strongly influence insurance pricing  
- A deployed **Streamlit app** for interactive prediction  

---

## 📊 Data
- **Dataset Link:** [Kaggle - Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download)  
- **Shape:** 986 rows × 11 columns  
- **Features:** 4 numerical, 7 categorical  
- **Null & Duplicates:** None  

---

## ⚙️ Method
- **Approach:** Supervised Regression  
- **Models Tested:**  
  - K-Nearest Regressor (KNR)  
  - Support Vector Regressor (SVR)  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
- **Best Model:** Random Forest Regressor (`BestRFR.pkl`)  

---

## 🛠️ Tech Stacks

### 🔹 Languages
- Python  
- Pandas  
- NumPy  

### 🔹 Tools
- Jupyter Notebook  
- Streamlit  
- Scikit-learn  
- Pickle  

### 🔹 Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kendalltau

import pickle
from utils import replaceBMI
```

## 📖 References
- [NCBI - The Cost of Cancer Care](https://khalinsurancepredict-029.streamlit.app/)
- [KFF - Americans’ Challenges with Health Care Costs](https://www.ncbi.nlm.nih.gov/books/NBK223643/)
- [Healthcare Cost](https://www.kff.org/health-costs/issue-brief/americans-challenges-with-health-care-costs/)



