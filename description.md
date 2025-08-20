# Milestone 2 - Predict Insurance (Annual) Price

## Repository Outline
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
    ├── description.md
    ├── P1M2_Khalif.ipynb
    ├── P1M2_Khalif_Inference.ipynb
    └── Medicalpremium.pcsv
    ├── conceptual.txt
    └── README.md
    └── BestRFR.pkl
    └── utils.py
    ├── url.txt
```

## Problem Background

Banyak orang menganggap remeh betapa mahalnya biaya perawatan kesehatan. Yang ternyata di US satu penyakit seperti kanker atau operasi jantung bisa menghabiskan biaya lebih dari $100.000, sehingga memaksa keluarga untuk menguras tabungan, menjual rumah, atau bahkan menjadi bangkrut. Dengan menggunakan asuransi, itu melindungi keluarga dengan menanggung sebagian besar biaya ini. Walaupun premi tetap mengacu pada faktor pribadi seperti usia, BMI, kebiasaan merokok, dan jumlah tanggungannya. 

Untuk itu saya akan membantu orang untuk memprediksi harga premi tahunan sehingga suatu keluarga bisa memperkirakan biaya asuransi mereka berdasarkan faktor-faktor ini, sehingga mereka dapat merencanakan ke depan dan menghindari bencana finansial akibat tagihan medis yang tak terduga.

## Project Output

Memberikan Prediksi Terkait Harga Premi Tahunan Yang Dipengaruhi Oleh Beberapa Kondisi Yang Dialami Individu.

## Data
Dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download
Dataset ini adalah Informasi Existing Harga Premi Asuransi.
Shape Dataset: 11 Columns , 986 Rows
Type Columns: 4 Numerikal Dan 7 Kategorikal
Null & Duplicated Values  : 0

## Method
1. Supervised : Regression
2. Model: KNR, SVR, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor 

## Stacks
1. import pandas as pd
2. import numpy as np
3. import matplotlib.pyplot as plt
4. import seaborn as sns
5. from sklearn.model_selection import train_test_split
6. from statsmodels.stats.outliers_influence import variance_inflation_factor
7. from scipy.stats import kendalltau
8. from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
9. from sklearn.preprocessing import OneHotEncoder
10. from sklearn.compose import ColumnTransformer
11. from sklearn.pipeline import Pipeline
12. from sklearn.model_selection import cross_val_score, GridSearchCV
13. from sklearn.metrics import make_scorer, mean_absolute_percentage_error,mean_absolute_error
14. import pickle
15. from sklearn.neighbors import KNeighborsRegressor
16. from sklearn.svm import SVR
17. from sklearn.tree import DecisionTreeRegressor
18. from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
19. from utils import replaceBMI

## Reference
1. https://khalinsurancepredict-029.streamlit.app/
2. https://www.ncbi.nlm.nih.gov/books/NBK223643/
3. https://www.kff.org/health-costs/issue-brief/americans-challenges-with-health-care-costs/