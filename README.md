# Detec-o_de_Fraudes_em_Transa-es
* Fazendo Detecção de Fraudes com Bibliotecas do Python

https://colab.research.google.com/drive/1Z4XtOpOTpaKrMJeCJru5ulbJb3CIlZ_c#scrollTo=_f5ADyHwxiUp

## Primeiro importo a biblioteca Pandas

```import pandas as pd

url = "https://www.dropbox.com/s/b44o3t3ehmnx2b7/creditcard.csv?dl=1"

df = pd.read_csv(url)
df.head()
```
    
## Problema de Clacificação Desbalanceada

```df['Class'].value_counts(normalize=True)```

## **Feature Engineering**

```import numpy as np
df['Amount_log'] = np.log1p(df['Amount'])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Time_scaled', 'Amount_scaled']] = scaler.fit_transform(df[['Time', 'Amount_log']])

from sklearn.model_selection import train_test_split
X = df.drop('Class', axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True)
```

## Logistic Regression

```from sklearn.linear_model import LogisticRegression

# Set random seed for reproducibility
np.random.seed(2)

# Initialize and train the Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(x_train, y_train)

# Make dredictions on the test set
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)
```

```from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
```

## **Balanceamento de dados**

```from imblearn.under_sampling import RandomUnderSampler

# Instantiate RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Apply RandomUnderSampler to the training data
x_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)

print("Class distribution after RandomUnderSampler:\n")
print(y_train_resampled.value_counts(normalize=True))
```

```from imblearn.over_sampling import SMOTE

# Instantiate SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

print("Class distribution after SMOTE:\n")
print(y_train_resampled.value_counts(normalize=True))
```
## Random

```from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf.fit(x_train_resampled, y_train_resampled)

y_pred = rf.predict(x_test)
y_proba = rf.predict_proba(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## Pipeline

```from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
y_proba = pipeline.predict_proba(x_test)
```

```thredhold = 0.3
y_pred = np.where(y_proba[:, 1] > thredhold, 1, 0)

print(classification_report(y_test, y_pred))
```
