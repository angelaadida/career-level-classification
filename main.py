import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTEN


def filter_location(loc):
    result = re.findall(r',\s[A-Z]{2}$', loc)
    return result[0][2:] if len(result) else loc


# 1. LOAD DATA
data = pd.read_excel('06_data_career.ods', engine='odf', dtype=str)
data = data.dropna(axis=0)
print(f'Shape: {data.shape}')
print(data['career_level'].value_counts())

# 2. CLEAN LOCATION
data['location'] = data['location'].apply(filter_location)

# 3. SPLIT
target = 'career_level'
X = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. SMOTEN OVERSAMPLING
print('\nBefore:', y_train.value_counts().to_dict())
ros = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy={
    'managing_director_small_medium_company': 500,
    'specialist': 500,
    'director_business_unit_leader': 500,
    'bereichsleiter': 1000
})
X_train, y_train = ros.fit_resample(X_train, y_train)
print('After:', y_train.value_counts().to_dict())

# 5. PIPELINE
preprocessor = ColumnTransformer(transformers=[
    ('title',       TfidfVectorizer(), 'title'),
    ('location',    OneHotEncoder(handle_unknown='ignore'), ['location']),
    ('description', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=0.01, max_df=0.95), 'description'),
    ('function',    OneHotEncoder(handle_unknown='ignore'), ['function']),
    ('industry',    TfidfVectorizer(), 'industry'),
])

pipeline = Pipeline(steps=[
    ('preprocessor',      preprocessor),
    ('feature_selection', SelectPercentile(chi2, percentile=5)),
    ('model',             RandomForestClassifier(random_state=42, n_jobs=-1))
])

# 6. TRAIN & EVALUATE
print('\nTraining...')
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. CONFUSION MATRIX
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix — Career Level Classification')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
