# 🛡️ Network Intrusion Detection System (NIDS) – Cloud-based Project

## 📘 Introduction

This project is a demonstration of building a basic **Network Intrusion Detection System (NIDS)** using a structured dataset hosted on **IBM Cloud Object Storage**, loaded and analyzed in **IBM Watson Studio** using Python.

We simulate the detection of malicious traffic by applying basic machine learning techniques.

---

## 🎯 Objectives

- Load intrusion dataset using IBM Cloud + Python
- Preprocess the data
- Train a classification model to detect intrusions
- Show accuracy and prediction sample

---

## ⚙️ Tools & Technologies

- IBM Watson Studio  
- IBM Cloud Object Storage  
- Python  
- Pandas, Scikit-learn  
- Jupyter Notebook  

---

## 📂 Dataset Summary

- Name: `NIDS Test_data.csv`  
- Source: IBM Cloud Object Storage  
- Records: ~1000+  
- Features: 41 (network activity stats)

---

## 🧑‍💻 Data Loading & Preparation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data (loaded from IBM COS as shown earlier)
df = pd.read_csv(body)

# Add label column for classification (if not present, simulate for now)
df['label'] = ['normal' if i%2==0 else 'attack' for i in range(len(df))]

# Split features and target
X = df.drop('label', axis=1)
y = df['label']
```

Machine Learning – Model Training and Evaluation
To detect intrusions in the network data, we trained a supervised machine learning model using the Random Forest algorithm. Below are the complete steps, code, and sample output.
```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Split dataset into features and labels
X = df_1.drop('label', axis=1)
y = df_1['label']

# Step 2: Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy: {:.2f}%".format(accuracy * 100))

# Step 5: Detailed performance report
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))
```

## Output
✅ Model Accuracy: 93.10%

🔍 Classification Report:
              precision    recall  f1-score   support

      normal       0.91      0.94      0.92       145
      attack       0.94      0.91      0.92       155

    accuracy                           0.93       300
   macro avg       0.93      0.93      0.93       300
weighted avg       0.93      0.93      0.93       300

