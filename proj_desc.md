For your project **"Learning Analytics to Predict Learner Performance in Early Childhood Education"**, the **xAPI-Edu-Data** dataset is one of the best choices. It contains student demographics, learning behavior (raising hands, visiting resources, discussions, etc.), and performance labels, making it suitable for learning analytics research. ([Kaggle][1])

### Dataset

Download it from:

* [Kaggle: xAPI-Edu-Data Dataset](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data/data?utm_source=chatgpt.com)

The file you need is:

```
xAPI-Edu-Data.csv
```

Upload this CSV file to your Google Colab session.

---

# Google Colab Code

### Step 1: Import Libraries

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

### Step 2: Upload Dataset

```python
from google.colab import files

uploaded = files.upload()
```

Select

```
xAPI-Edu-Data.csv
```

---

### Step 3: Read Dataset

```python
df = pd.read_csv("xAPI-Edu-Data.csv")

df.head()
```

---

### Step 4: Dataset Information

```python
df.info()
```

```python
df.describe()
```

```python
df.isnull().sum()
```

---

### Step 5: Visualize Class Distribution

```python
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)

plt.title("Student Performance Classes")
plt.show()
```

---

### Step 6: Encode Categorical Variables

```python
encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = encoder.fit_transform(df[column])
```

---

### Step 7: Split Features and Target

```python
X = df.drop("Class", axis=1)

y = df["Class"]
```

---

### Step 8: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### Step 9: Train Random Forest Model

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
```

---

### Step 10: Prediction

```python
y_pred = model.predict(X_test)
```

---

### Step 11: Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

Expected accuracy is usually around **80–90%**, depending on the train/test split and preprocessing. ([ResearchGate][2])

---

### Step 12: Classification Report

```python
print(classification_report(y_test, y_pred))
```

---

### Step 13: Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            cmap="Blues",
            fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---

### Step 14: Feature Importance

```python
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance")
plt.show()
```

---

## Expected Output

You will obtain:

* ✔ Dataset overview
* ✔ Data preprocessing
* ✔ Label encoding
* ✔ Random Forest model
* ✔ Accuracy score
* ✔ Precision, Recall, F1-score
* ✔ Confusion Matrix
* ✔ Feature Importance graph

---

## Project Title

**Learning Analytics to Predict Learner Performance in Early Childhood Education Using Machine Learning**

### Algorithms You Can Compare

For a stronger research paper, compare multiple models:

| Algorithm              | Expected Accuracy |
| ---------------------- | ----------------- |
| Decision Tree          | 75–82%            |
| Random Forest          | 82–90%            |
| Logistic Regression    | 72–80%            |
| Support Vector Machine | 80–88%            |
| K-Nearest Neighbors    | 75–85%            |
| XGBoost                | 88–93%            |

Using multiple algorithms and comparing their performance will make your project and research paper more substantial.

[1]: https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data/data?utm_source=chatgpt.com "Students' Academic Performance Dataset (xAPI-Edu-Data)"
[2]: https://www.researchgate.net/figure/Student-performance-analysis-on-xAPI-Edu-Data_fig4_359899905?utm_source=chatgpt.com "Student performance analysis on xAPI-Edu-Data."
